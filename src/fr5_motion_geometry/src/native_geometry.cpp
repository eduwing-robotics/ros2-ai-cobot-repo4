// A task-scoped CPU query process. No ROS init, executor, node, service or action.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <rclcpp/serialization.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>

using Json = nlohmann::json;
constexpr std::size_t MAX_LINE_BYTES = 16 * 1024 * 1024;
constexpr std::size_t MAX_CONTACTS = 100000;
constexpr std::size_t MAX_CONTACTS_PER_PAIR = 1000;
const std::set<std::string> QUERY_JOINTS = {"j1", "j2", "j3", "j4", "j5", "j6", "finger_right_joint"};

void require(bool condition, const char* code)
{
  if (!condition) throw std::runtime_error(code);
}

void keys(const Json& value, std::initializer_list<const char*> expected)
{
  require(value.is_object() && value.size() == expected.size(), "SCHEMA");
  for (const auto* key : expected) require(value.contains(key), "SCHEMA");
}

bool valid_id(const Json& id)
{
  return id.is_number_integer() && !id.is_boolean() && id > 0 && id <= 9007199254740991ULL;
}

template<class Message> std::string encode(const Message& message)
{
  rclcpp::SerializedMessage serialized;
  rclcpp::Serialization<Message>().serialize_message(&message, &serialized);
  const auto& raw = serialized.get_rcl_serialized_message();
  require(raw.buffer_length <= MAX_LINE_BYTES / 2, "PAYLOAD_LIMIT");
  constexpr char HEX[] = "0123456789abcdef";
  std::string result(raw.buffer_length * 2, '0');
  for (std::size_t i = 0; i < raw.buffer_length; ++i)
  {
    result[2*i] = HEX[raw.buffer[i] >> 4];
    result[2*i+1] = HEX[raw.buffer[i] & 15];
  }
  return result;
}

template<class Message> Message decode(const Json& value)
{
  require(value.is_string(), "CDR_HEX");
  const auto& hex = value.get_ref<const std::string&>();
  require(hex.size() >= 8 && hex.size() <= MAX_LINE_BYTES && hex.size() % 2 == 0, "CDR_HEX");
  rclcpp::SerializedMessage serialized(hex.size() / 2);
  auto& raw = serialized.get_rcl_serialized_message();
  auto nibble = [](char c) -> unsigned {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    throw std::runtime_error("CDR_HEX");
  };
  for (std::size_t i = 0; i < hex.size()/2; ++i) raw.buffer[i] = nibble(hex[2*i])*16 + nibble(hex[2*i+1]);
  raw.buffer_length = hex.size()/2;
  Message message;
  try { rclcpp::Serialization<Message>().deserialize_message(&serialized, &message); }
  catch (const std::exception&) { throw std::runtime_error("CDR_INVALID"); }
  // CDR padding is not canonical across Python/C++ serializers. The native
  // roundtrip must consume the same extent; padding bytes need not be equal.
  require(encode(message).size() == hex.size(), "CDR_FRAMING");
  return message;
}

void pose(const geometry_msgs::msg::Pose& value)
{
  const auto& p = value.position;
  const auto& q = value.orientation;
  require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
          std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w) &&
          std::abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) <= 1e-9, "GEOMETRY_POSE");
}

void geometry(const moveit_msgs::msg::CollisionObject& object)
{
  require(!object.id.empty() && !object.header.frame_id.empty() &&
          object.operation == object.ADD, "GEOMETRY_OBJECT");
  pose(object.pose);
  require(object.primitives.size() == object.primitive_poses.size() &&
          object.meshes.size() == object.mesh_poses.size() && object.planes.size() == object.plane_poses.size() &&
          (!object.primitives.empty() || !object.meshes.empty() || !object.planes.empty()), "GEOMETRY_SHAPES");
  for (const auto& shape : object.primitives)
  {
    std::size_t count = shape.type == shape.BOX ? 3 : shape.type == shape.SPHERE ? 1 :
                        (shape.type == shape.CYLINDER || shape.type == shape.CONE) ? 2 : 0;
    require(count && shape.dimensions.size() == count, "GEOMETRY_SHAPES");
    for (auto value : shape.dimensions) require(std::isfinite(value) && value > 0, "GEOMETRY_SHAPES");
  }
  for (const auto& mesh : object.meshes)
  {
    require(!mesh.vertices.empty() && !mesh.triangles.empty(), "GEOMETRY_SHAPES");
    for (const auto& p : mesh.vertices) require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), "GEOMETRY_SHAPES");
    for (const auto& triangle : mesh.triangles)
      for (auto index : triangle.vertex_indices) require(index < mesh.vertices.size(), "GEOMETRY_SHAPES");
  }
  for (const auto& plane : object.planes)
  {
    for (auto value : plane.coef) require(std::isfinite(value), "GEOMETRY_SHAPES");
    require(plane.coef[0]*plane.coef[0]+plane.coef[1]*plane.coef[1]+plane.coef[2]*plane.coef[2] > 0, "GEOMETRY_SHAPES");
  }
  for (const auto& p : object.primitive_poses) pose(p);
  for (const auto& p : object.mesh_poses) pose(p);
  for (const auto& p : object.plane_poses) pose(p);
  require(object.subframe_names.size() == object.subframe_poses.size() &&
          std::set<std::string>(object.subframe_names.begin(), object.subframe_names.end()).size() == object.subframe_names.size(), "GEOMETRY_SUBFRAMES");
  for (const auto& name : object.subframe_names) require(!name.empty(), "GEOMETRY_SUBFRAMES");
  for (const auto& p : object.subframe_poses) pose(p);
}

std::size_t shape_count(const moveit_msgs::msg::CollisionObject& object)
{
  return object.primitives.size() + object.meshes.size() + object.planes.size();
}

void state_message(const moveit_msgs::msg::RobotState& message, const moveit::core::RobotModel& model, bool query)
{
  if (query) require(message.is_diff, "STATE_DIFF_REQUIRED");
  const auto& joints = message.joint_state;
  require(joints.name.size() == joints.position.size(), "STATE_JOINTS");
  std::set<std::string> names;
  const auto& known = model.getVariableNames();
  for (std::size_t i = 0; i < joints.name.size(); ++i)
    require(names.insert(joints.name[i]).second && std::find(known.begin(), known.end(), joints.name[i]) != known.end() &&
            std::isfinite(joints.position[i]), "STATE_JOINTS");
  if (query) require(names == QUERY_JOINTS, "STATE_JOINTS");
  else for (const auto& name : QUERY_JOINTS) require(names.count(name), "STATE_JOINTS");
  for (const auto* values : {&joints.velocity, &joints.effort})
  {
    require(values->empty() || values->size() == joints.name.size(), "STATE_JOINTS");
    for (auto v : *values) require(std::isfinite(v), "STATE_JOINTS");
  }
  const auto& multi = message.multi_dof_joint_state;
  require(multi.joint_names.empty() && multi.transforms.empty() && multi.twist.empty() && multi.wrench.empty(), "STATE_JOINTS");
  std::set<std::string> ids;
  for (const auto& body : message.attached_collision_objects)
  {
    geometry(body.object);
    require(ids.insert(body.object.id).second && model.hasLinkModel(body.link_name) &&
            body.object.header.frame_id == body.link_name && !model.hasLinkModel(body.object.id), "ATTACHMENT_GEOMETRY");
    for (const auto& link : body.touch_links) require(model.hasLinkModel(link), "ATTACHMENT_GEOMETRY");
  }
}

void check_converted_state(const moveit_msgs::msg::RobotState& message, const moveit::core::RobotState& state)
{
  for (const auto& body : message.attached_collision_objects)
  {
    const auto* actual = state.getAttachedBody(body.object.id);
    require(actual && actual->getAttachedLinkName() == body.link_name &&
            actual->getShapes().size() == shape_count(body.object), "ATTACHMENT_CONVERSION");
  }
  for (const auto* link : state.getRobotModel()->getLinkModels())
    require(state.getGlobalLinkTransform(link).matrix().allFinite(), "MODEL_TRANSFORM");
}

class Geometry
{
public:
  Json init(const Json& request)
  {
    keys(request, {"op", "id", "urdf", "srdf", "scene_cdr_hex"});
    require(!scene_, "ALREADY_INITIALIZED");
    require(request["urdf"].is_string() && request["srdf"].is_string(), "MODEL_SCHEMA");
    const auto urdf = urdf::parseURDF(request["urdf"].get<std::string>());
    require(bool(urdf), "MODEL_URDF");
    for (const auto& entry : urdf->links_) for (const auto& collision : entry.second->collision_array)
    {
      require(collision && collision->geometry, "MODEL_SHAPES");
      const auto& origin = collision->origin;
      geometry_msgs::msg::Pose p;
      p.position.x=origin.position.x; p.position.y=origin.position.y; p.position.z=origin.position.z;
      p.orientation.x=origin.rotation.x; p.orientation.y=origin.rotation.y;
      p.orientation.z=origin.rotation.z; p.orientation.w=origin.rotation.w; pose(p);
      std::vector<double> dimensions;
      const auto& shape = collision->geometry;
      if (shape->type == urdf::Geometry::BOX)
      {
        auto box = std::static_pointer_cast<urdf::Box>(shape);
        dimensions = {box->dim.x, box->dim.y, box->dim.z};
      }
      else if (shape->type == urdf::Geometry::SPHERE) dimensions = {std::static_pointer_cast<urdf::Sphere>(shape)->radius};
      else if (shape->type == urdf::Geometry::CYLINDER)
      {
        auto cylinder = std::static_pointer_cast<urdf::Cylinder>(shape);
        dimensions = {cylinder->radius, cylinder->length};
      }
      else if (shape->type == urdf::Geometry::MESH)
      {
        auto mesh = std::static_pointer_cast<urdf::Mesh>(shape);
        require(!mesh->filename.empty(), "MODEL_SHAPES");
        dimensions = {mesh->scale.x, mesh->scale.y, mesh->scale.z};
      }
      require(!dimensions.empty(), "MODEL_SHAPES");
      for (auto dimension : dimensions) require(std::isfinite(dimension) && dimension > 0, "MODEL_SHAPES");
    }
    auto srdf = std::make_shared<srdf::Model>();
    require(srdf->initString(*urdf, request["srdf"].get<std::string>()), "MODEL_SRDF");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    require(model->hasLinkModel("gripper_link"), "MODEL_LINK");
    std::size_t expected = 0, actual = 0;
    for (const auto& entry : urdf->links_) expected += entry.second->collision_array.size();
    for (const auto* link : model->getLinkModelsWithCollisionGeometry())
      for (const auto& shape : link->getShapes()) { require(bool(shape), "MODEL_SHAPES"); ++actual; }
    require(expected == actual && actual > 0, "MODEL_SHAPES");
    auto message = decode<moveit_msgs::msg::PlanningScene>(request["scene_cdr_hex"]);
    require(!message.is_diff && message.robot_model_name == model->getName(), "SCENE_FULL_REQUIRED");
    state_message(message.robot_state, *model, false);
    const auto& acm = message.allowed_collision_matrix;
    require(acm.entry_names.size() == acm.entry_values.size() &&
            acm.default_entry_names.size() == acm.default_entry_values.size(), "SCENE_ACM");
    require(std::set<std::string>(acm.entry_names.begin(), acm.entry_names.end()).size() == acm.entry_names.size() &&
            std::set<std::string>(acm.default_entry_names.begin(), acm.default_entry_names.end()).size() == acm.default_entry_names.size(), "SCENE_ACM");
    for (const auto& row : acm.entry_values) require(row.enabled.size() == acm.entry_names.size(), "SCENE_ACM");
    for (std::size_t i=0; i<acm.entry_names.size(); ++i) for (std::size_t j=0; j<i; ++j)
      require(acm.entry_values[i].enabled[j] == acm.entry_values[j].enabled[i], "SCENE_ACM");
    std::set<std::string> padded, scaled, frames;
    for (const auto& p : message.link_padding) require(padded.insert(p.link_name).second && model->hasLinkModel(p.link_name) && std::isfinite(p.padding) && p.padding >= 0, "SCENE_PADDING");
    for (const auto& s : message.link_scale) require(scaled.insert(s.link_name).second && model->hasLinkModel(s.link_name) && std::isfinite(s.scale) && s.scale > 0, "SCENE_PADDING");
    for (const auto& tf : message.fixed_frame_transforms)
    {
      geometry_msgs::msg::Pose p; p.position.x=tf.transform.translation.x; p.position.y=tf.transform.translation.y;
      p.position.z=tf.transform.translation.z; p.orientation=tf.transform.rotation; pose(p);
      // MoveIt fixed-frame messages name the source in header.frame_id and
      // the planning target in child_frame_id (not ordinary TF-tree edges).
      require(moveit::core::Transforms::sameFrame(tf.child_frame_id, model->getModelFrame()) &&
              !tf.header.frame_id.empty() && frames.insert(tf.header.frame_id).second, "SCENE_TRANSFORMS");
    }
    const auto& octomap = message.world.octomap;
    if (!octomap.octomap.data.empty())
    {
      pose(octomap.origin);
      require(octomap.octomap.id == "OcTree" && std::isfinite(octomap.octomap.resolution) &&
              octomap.octomap.resolution > 0 && !octomap.header.frame_id.empty(), "SCENE_OCTOMAP");
    }
    std::set<std::string> ids;
    for (const auto& body : message.robot_state.attached_collision_objects) ids.insert(body.object.id);
    for (const auto& object : message.world.collision_objects)
    {
      geometry(object);
      require(ids.insert(object.id).second && !model->hasLinkModel(object.id), "SCENE_OBJECT_ID");
    }
    auto scene = std::make_shared<planning_scene::PlanningScene>(model);
    require(scene->setPlanningSceneMsg(message), "SCENE_CONVERSION");
    if (!octomap.octomap.data.empty())
      require(scene->knowsFrameTransform(octomap.header.frame_id) &&
              scene->getWorld()->hasObject(planning_scene::PlanningScene::OCTOMAP_NS), "SCENE_OCTOMAP");
    for (const auto& tf : message.fixed_frame_transforms)
    {
      require(scene->getTransforms().isFixedFrame(tf.header.frame_id), "SCENE_TRANSFORMS");
      const auto& actual_tf = scene->getTransforms().getTransform(tf.header.frame_id);
      const auto& q = tf.transform.rotation;
      const auto& t = tf.transform.translation;
      require((actual_tf.translation() - Eigen::Vector3d(t.x,t.y,t.z)).norm() <= 1e-9 &&
              (actual_tf.linear() - Eigen::Quaterniond(q.w,q.x,q.y,q.z).toRotationMatrix()).norm() <= 1e-9, "SCENE_TRANSFORMS");
    }
    scene->getCurrentStateNonConst().update();
    check_converted_state(message.robot_state, scene->getCurrentState());
    for (const auto& object : message.world.collision_objects)
    {
      const auto actual_object = scene->getWorld()->getObject(object.id);
      require(scene->knowsFrameTransform(object.header.frame_id) && actual_object &&
              actual_object->shapes_.size() == shape_count(object), "SCENE_CONVERSION");
    }
    scene_ = std::move(scene);  // Only a completely validated init becomes owned.
    return {{"op", "init"}, {"id", request["id"]}, {"ok", true}};
  }

  Json query(const Json& request)
  {
    keys(request, {"op", "id", "variants"});
    require(bool(scene_), "NOT_INITIALIZED");
    require(request["variants"].is_array() && !request["variants"].empty() && request["variants"].size() <= 3, "QUERY_VARIANTS");
    Json variants = Json::array();
    std::set<std::string> hypotheses;
    std::size_t reply_bytes = 0;
    for (const auto& variant : request["variants"])
    {
      keys(variant, {"hypothesis", "world_objects_cdr_hex", "states_cdr_hex"});
      require(variant["hypothesis"].is_string(), "QUERY_VARIANTS");
      const auto hypothesis = variant["hypothesis"].get<std::string>();
      require((hypothesis == "source" || hypothesis == "carried" || hypothesis == "released") &&
              hypotheses.insert(hypothesis).second, "QUERY_VARIANTS");
      require(variant["world_objects_cdr_hex"].is_array() && variant["states_cdr_hex"].is_array() &&
              !variant["states_cdr_hex"].empty(), "QUERY_VARIANTS");
      auto local = planning_scene::PlanningScene::clone(scene_);
      for (const auto& encoded : variant["world_objects_cdr_hex"])
      {
        const auto object = decode<moveit_msgs::msg::CollisionObject>(encoded);
        geometry(object);
        require(!local->getWorld()->hasObject(object.id) && !local->getCurrentState().hasAttachedBody(object.id) &&
                !local->getRobotModel()->hasLinkModel(object.id), "WORLD_OBJECT_REPLACEMENT");
        require(local->knowsFrameTransform(object.header.frame_id), "GEOMETRY_FRAME");
        require(local->processCollisionObjectMsg(object), "GEOMETRY_CONVERSION");
        require(local->getWorld()->getObject(object.id) &&
                local->getWorld()->getObject(object.id)->shapes_.size() == shape_count(object), "GEOMETRY_CONVERSION");
      }
      Json samples = Json::array();
      for (const auto& encoded : variant["states_cdr_hex"])
      {
        const auto message = decode<moveit_msgs::msg::RobotState>(encoded);
        state_message(message, *local->getRobotModel(), true);
        for (const auto& body : message.attached_collision_objects)
          require(!local->getCurrentState().hasAttachedBody(body.object.id) &&
                  !local->getWorld()->hasObject(body.object.id), "ATTACHMENT_REPLACEMENT");
        auto state = local->getCurrentState();  // No previous query/sample additions.
        require(moveit::core::robotStateMsgToRobotState(message, state), "STATE_CONVERSION");
        state.update();
        check_converted_state(message, state);
        collision_detection::CollisionRequest req;
        req.group_name = ""; req.contacts = true;
        req.max_contacts = MAX_CONTACTS; req.max_contacts_per_pair = MAX_CONTACTS_PER_PAIR;
        collision_detection::CollisionResult result;
        local->checkCollision(req, result, state);
        bool saturated = result.contact_count >= MAX_CONTACTS;
        moveit_msgs::srv::GetStateValidity::Response response;
        response.valid = !result.collision;
        for (const auto& pair : result.contacts)
        {
          saturated = saturated || pair.second.size() >= MAX_CONTACTS_PER_PAIR;
          for (const auto& c : pair.second)
          {
            moveit_msgs::msg::ContactInformation contact;
            contact.header.frame_id = local->getPlanningFrame();
            contact.contact_body_1 = c.body_name_1; contact.contact_body_2 = c.body_name_2;
            auto body_type = [](collision_detection::BodyType type) -> uint32_t {
              switch (type) {
                case collision_detection::BodyTypes::ROBOT_LINK: return moveit_msgs::msg::ContactInformation::ROBOT_LINK;
                case collision_detection::BodyTypes::ROBOT_ATTACHED: return moveit_msgs::msg::ContactInformation::ROBOT_ATTACHED;
                case collision_detection::BodyTypes::WORLD_OBJECT: return moveit_msgs::msg::ContactInformation::WORLD_OBJECT;
              }
              throw std::runtime_error("CONTACT_BODY_TYPE");
            };
            contact.body_type_1=body_type(c.body_type_1); contact.body_type_2=body_type(c.body_type_2);
            contact.depth=c.depth;
            contact.position.x=c.pos.x(); contact.position.y=c.pos.y(); contact.position.z=c.pos.z();
            contact.normal.x=c.normal.x(); contact.normal.y=c.normal.y(); contact.normal.z=c.normal.z();
            require(c.pos.allFinite() && c.normal.allFinite() && c.normal.squaredNorm() > 0 &&
                    std::isfinite(c.depth) && c.depth >= 0, "CONTACT_GEOMETRY");
            response.contacts.push_back(contact);
          }
        }
        const auto& fk = state.getGlobalLinkTransform("gripper_link");
        Json columns = Json::array();
        for (int c=0; c<3; ++c) columns.push_back({fk.linear()(0,c), fk.linear()(1,c), fk.linear()(2,c)});
        Json sample = {{"response_cdr_hex", encode(response)}, {"saturated", saturated},
                       {"gripper_pose", {{"translation_m", {fk.translation().x(),fk.translation().y(),fk.translation().z()}},
                                         {"rotation_columns", columns}}}};
        reply_bytes += sample.dump().size();
        require(reply_bytes <= MAX_LINE_BYTES, "PAYLOAD_LIMIT");
        samples.push_back(std::move(sample));
      }
      variants.push_back({{"hypothesis", hypothesis}, {"samples", std::move(samples)}});
    }
    return {{"op", "query"}, {"id", request["id"]}, {"ok", true}, {"variants", std::move(variants)}};
  }
private:
  planning_scene::PlanningScenePtr scene_;
};

int main()
{
  Geometry geometry;
  std::string line;
  while (std::cin.peek() != std::char_traits<char>::eof())
  {
    line.clear(); bool oversized = false; char ch;
    while (std::cin.get(ch) && ch != '\n')
    {
      if (line.size() < MAX_LINE_BYTES) line.push_back(ch);
      else oversized = true;
    }
    Json op = "", id = 0, response;
    try
    {
      require(!oversized, "PAYLOAD_LIMIT");
      std::vector<std::set<std::string>> object_keys;
      auto callback = [&object_keys](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) object_keys.emplace_back();
        else if (event == Json::parse_event_t::key) require(object_keys.back().insert(value.get<std::string>()).second, "DUPLICATE_KEY");
        else if (event == Json::parse_event_t::object_end) object_keys.pop_back();
        return true;
      };
      const auto request = Json::parse(line, callback);
      if (request.is_object() && request.contains("op") && request["op"].is_string()) op = request["op"];
      if (request.is_object() && request.contains("id") && valid_id(request["id"])) id = request["id"];
      require(id != 0 && op.is_string(), "SCHEMA");
      if (op == "init") response = geometry.init(request);
      else if (op == "query") response = geometry.query(request);
      else throw std::runtime_error("SCHEMA");
      require(response.dump().size() <= MAX_LINE_BYTES, "PAYLOAD_LIMIT");
    }
    catch (const nlohmann::json::exception&) { response={{"op",op},{"id",id},{"ok",false},{"error","JSON_SCHEMA"}}; }
    catch (const std::exception& error) { response={{"op",op},{"id",id},{"ok",false},{"error",error.what()}}; }
    std::cout << response.dump() << '\n' << std::flush;
  }
}
