// CPU fixture only: native methods are inserted from the tracked vendor patch.
// No SDK linkage, ROS node, socket, controller or physical completion.
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "gripper_execution_evidence.hpp"
#define RCLCPP_INFO(...) ((void)0)
#define RCLCPP_WARN(...) ((void)0)
#define RCLCPP_ERROR(...) ((void)0)
namespace rclcpp {struct Time {};struct Duration {};}
namespace hardware_interface {enum class return_type {OK, ERROR};}
namespace fairino_hardware {
std::atomic<int> udp_command_error{0};
// The optional candidate-SDK fixture supplies these exact public types.
#ifndef FR5_SNAPSHOT_SDK_TEST
struct JointPos {double jPos[6]{};};
struct ExaxisPos {double values[4];};
struct ROBOT_STATE_PKG {
  struct {int year=1970,mouth=1,day=1,hour=0,minute=0,second=10,millisecond=0;} robotTime;
  uint8_t frame_cnt=1,gripper_position=100,gripper_motiondone=0;
  int gripper_fault=0,main_code=0,sub_code=0;
  double jt_cur_pos[6]{};
};
struct RobotStateSnapshot {
  ROBOT_STATE_PKG state{}; int64_t host_receive_steady_ns=0;
  uint64_t producer_sequence=0, connection_epoch=0, configuration_epoch=0; bool valid=false;
};
constexpr int ROBOT_SNAPSHOT_OK=0;
#endif
struct Robot {
  int GetRobotRealTimeStateSnapshot(RobotStateSnapshot *p) {
#ifdef FR5_SNAPSHOT_SDK_TEST
    return injected_snapshot(p);
#else
    (void)p; return -1;
#endif
  }
  int moves=0,resumes=0,target=100,polls=0,move_error=0,resume_error=0,arm_sends=0;
  JointPos last_arm;
  bool sampler_only=false;
  std::thread::id sampler_thread;
  bool stale=false,settled=false,clock_mode=false,delayed=false;
  int delay_ms=10;
  int terminal_scenario=0;
  std::chrono::steady_clock::time_point move_at;
  std::function<void()> initial_hook,move_hook,poll_hook,resume_hook;
  ROBOT_STATE_PKG state;
  void source_now() {
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count()-3;
    std::time_t seconds=ms/1000;std::tm calendar{};gmtime_r(&seconds,&calendar);
    state.robotTime.year=calendar.tm_year+1900;state.robotTime.mouth=calendar.tm_mon+1;state.robotTime.day=calendar.tm_mday;
    state.robotTime.hour=calendar.tm_hour;state.robotTime.minute=calendar.tm_min;state.robotTime.second=calendar.tm_sec;state.robotTime.millisecond=int(ms%1000);
  }
  int GetRobotRealTimeState(ROBOT_STATE_PKG *p) {
    if(sampler_only) assert(std::this_thread::get_id()==sampler_thread);
    if (!moves) { if(clock_mode)source_now();if(initial_hook){auto f=std::move(initial_hook);initial_hook=nullptr;f();} }
    else {
      ++polls; state.gripper_position=target+(settled?1:0);state.gripper_motiondone=settled?0:1;
      if(!stale && !(delayed && polls==1)){if(clock_mode)source_now();else state.robotTime.millisecond=100+polls;state.frame_cnt=2;}
      if(poll_hook){auto f=std::move(poll_hook);poll_hook=nullptr;f();}
    }
    if(!moves && terminal_scenario==4) state.gripper_position=99;
    if(moves && terminal_scenario) {
      if(terminal_scenario==1 || terminal_scenario==2) state.gripper_position=100;
      if(terminal_scenario==2) state.gripper_motiondone=(clock_mode ? std::chrono::steady_clock::now()-move_at<std::chrono::milliseconds(40) : polls<3)?0:1;
      if(terminal_scenario==4) {state.gripper_position=99;state.gripper_motiondone=1;}
      if(terminal_scenario==3) {state.gripper_position=80;state.gripper_motiondone=1;}
    }
    *p=state;return 0;
  }
  std::array<int,10> sdk_tuple{};
  int MoveGripper(int index,int pos,int velocity,int force,int max_time,int a5,int a6,int a7,int a8,int a9) {
    sdk_tuple={index,pos,velocity,force,max_time,a5,a6,a7,a8,a9};
    ++moves;move_at=std::chrono::steady_clock::now();target=pos;if(move_hook)move_hook();
    if(clock_mode)std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    return move_error;
  }
  int ServoMoveStart(int) {++resumes;if(resume_hook)resume_hook();return resume_error;}
  int ServoJ(JointPos *value,ExaxisPos *,int,int,double,int,int,int,int){++arm_sends;last_arm=*value;return 0;}
};
struct Result {int send_error=0,controller_error=0;bool reply_received=true;bool ok()const{return send_error==0;}};
template<class F> Result send_udp_command_and_observe(F f){return {f(),0,true};}
struct Parameter {std::vector<double> values;std::vector<double> as_double_array()const{return values;}};
struct Node {Parameter parameter;Parameter get_parameter(const char *)const{return parameter;}};
struct PreciseControllerClock {
  std::atomic<int> delay_ms{0};
  std::vector<int> delays_ms; // Set before the worker starts; never mutated live.
  std::atomic<size_t> queries{0};
  double read(long timeout, const std::function<bool()> &cancel) {
    const auto index=queries.fetch_add(1);
    const int delay=delays_ms.empty() ? delay_ms.load() : delays_ms[index%delays_ms.size()];
    auto start=std::chrono::steady_clock::now();
    while(std::chrono::steady_clock::now()-start<std::chrono::milliseconds(delay)) {
      if(cancel()) throw std::runtime_error("cancelled");
      if(std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(timeout)) throw std::runtime_error("timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if(cancel()) throw std::runtime_error("cancelled");
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
};
struct FairinoHardwareInterface {
  std::thread _gripper_thread;
  void stop_gripper_worker();
  std::mutex _gripper_mutex;
  std::condition_variable _gripper_cv;
  bool _stop_gripper_thread=false;
  std::optional<int> _pending_gripper_position{56};
  std::atomic<uint64_t> _gripper_command_generation{1};
  std::atomic<bool> _gripper_rpc_active{false},_arm_stream_paused{true};
  std::atomic<int> _gripper_error{0};
  std::atomic<double> _gripper_position_state{.021},_last_gripper_command{.01176};
  double _gripper_upper_position=.021;
  int _gripper_index=1,_gripper_velocity=20,_gripper_open_velocity=20,_gripper_force=50,_gripper_open_force=50;
  int _gripper_max_time=1000,_gripper_settle_time_ms=0;
  std::unique_ptr<Robot> _ptr_robot=std::make_unique<Robot>();
  GripperExecutionEvidence _gripper_evidence;
  bool _require_gripper_source_clock=false;
  bool _require_coherent_snapshot=false;
  CoherentDelivery _delivery;
  std::array<double,2> _completion_epochs{};
  void sample_coherent_evidence();
  std::vector<double> _jnt_position_state=std::vector<double>(7,0.);
  bool _has_arm=true,_is_gripper=true;
  int _control_mode=0;
  std::array<size_t,6> _arm_joint_indices{0,1,2,3,4,5};
  size_t _gripper_joint_index=6;
  std::vector<double> _jnt_position_command{.01,-.02,.03,-.04,.05,-.06,.01176};
  std::vector<double> _jnt_torque_command=std::vector<double>(7,0.);
  GripperSourceClock _gripper_source_clock;
  std::unique_ptr<PreciseControllerClock> _precise_clock;
  std::array<double,35> _gripper_raw_sample{};
  std::array<double,92> _gripper_current_sample{};
  ROBOT_STATE_PKG _gripper_raw_state{}, _gripper_current_state{};
  double _last_controller_clock=0.;
  std::vector<double> _gripper_temporal_policy;
  bool refresh_gripper_freshness(double after_steady=0.);
  bool certified_gripper_observation(ROBOT_STATE_PKG &, std::array<double,92> &, double);
  std::shared_ptr<Node> node=std::make_shared<Node>();
  auto get_node(){return node;}
  std::array<double,GripperExecutionEvidence::names.size()> _gripper_evidence_values{};
  void gripper_worker();
  void sample_gripper_evidence();
  bool gripper_release_ready();
  hardware_interface::return_type write(const rclcpp::Time &,const rclcpp::Duration &);
};
// NATIVE_METHODS
}
int main(int argc,char **argv) {
  using namespace fairino_hardware;
  assert(argc==2 || argc==3);std::string mode=argv[1];FairinoHardwareInterface h;
  auto &r=*h._ptr_robot;h._gripper_evidence.activate();
  if(mode=="overlap_write") {
    // Source-only discriminator for an isolated overlap candidate. The fake
    // SDK counts calls; it makes no claim that physical overlap is supported.
    h._require_gripper_source_clock=true;h._require_coherent_snapshot=true;
    h._pending_gripper_position.reset();h._gripper_command_generation=0;
    h._arm_stream_paused=false;h._jnt_position_command[6]=.012;
    auto &v=h._gripper_evidence_values;
    const auto steady=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    v[0]=5.;v[27]=1.;v[42]=.3;v[10]=v[106]=v[110]=steady;
    v[107]=v[108]=v[109]=1.;
    assert(h.write({}, {})==hardware_interface::return_type::OK);
    const int initial_sends=r.arm_sends;
    assert(h._gripper_command_generation==1 && h._pending_gripper_position);
    // Simulate in-flight gripper state, with still-current coherent telemetry.
    h._pending_gripper_position.reset();h._gripper_rpc_active=true;
    h._gripper_evidence.active=1;v[5]=v[6]=1.;v[23]=1.;v[24]=h._arm_stream_paused?0.:1.;
    h._jnt_position_command[0]+=.01;
    assert(h.write({}, {})==hardware_interface::return_type::OK);
    const int active_sends=r.arm_sends;
    // Existing fault handling must survive the overlap candidate as well.
    udp_command_error=17;
    assert(h.write({}, {})==hardware_interface::return_type::ERROR);
    assert(r.arm_sends==active_sends && r.moves==0 && r.resumes==0);
    std::cout<<"{\"initial_arm_sends\":"<<initial_sends
             <<",\"active_arm_sends\":"<<active_sends
             <<",\"fault_stops_sends\":true,\"physical_overlap\":\"UNKNOWN\"}";
    return 0;
  }
  if(mode=="distinct_reference" || mode=="legacy_deadband") {
    h._require_gripper_source_clock=mode=="distinct_reference";
    h._pending_gripper_position.reset();h._gripper_command_generation=0;
    h._arm_stream_paused=false;h._jnt_position_command[6]=.01177;
    assert(h.write({}, {})==hardware_interface::return_type::OK);
    if(mode=="legacy_deadband") {
      assert(h._gripper_command_generation==0 && !h._pending_gripper_position && r.arm_sends==1);
    } else {
      assert(h._gripper_command_generation==1 && h._pending_gripper_position && r.arm_sends==0);
      assert(h._last_gripper_command==.01177);
      assert(h.write({}, {})==hardware_interface::return_type::OK); // same reference, no restart
      assert(h._gripper_command_generation==1 && r.arm_sends==0);
      h._jnt_position_command[6]=.01178;
      assert(h.write({}, {})==hardware_interface::return_type::ERROR);
      assert(h._gripper_command_generation==1 && h._last_gripper_command==.01177 && r.arm_sends==0);
    }
    std::cout<<"{\"native_write_only\":true}";return 0;
  }
  const bool guarded=mode.find("clock_")==0;
  if(guarded){
    assert(argc==3);h._require_gripper_source_clock=true;r.clock_mode=true;
    h._gripper_evidence.incarnation={1,2,3,4};h._gripper_max_time=120;
    std::istringstream input(argv[2]);std::string value;
    while(std::getline(input,value,','))h.node->parameter.values.push_back(std::stod(value));
    if(mode=="clock_cached")r.stale=true;
    if(mode=="clock_delayed_fresh")r.delayed=true;
    if(mode=="clock_expired_move" || mode=="clock_rebind_expired"){
      h.node->parameter.values[9]=h.node->parameter.values[7]+.02;r.delay_ms=40;
    }
    if(mode=="clock_rebind_expired")r.move_hook=[&]{h.node->parameter.values[9]+=10.;};
    if(mode=="clock_late_resume"){
      h.node->parameter.values[9]=h.node->parameter.values[7]+.12;
      r.resume_hook=[]{std::this_thread::sleep_for(std::chrono::milliseconds(140));};
    }
    if(mode=="clock_wrong_incarnation")h.node->parameter.values[1]=99.;
    if(mode=="clock_fractional_incarnation")h.node->parameter.values[1]+=.25;
    if(mode=="clock_oversized_incarnation")h.node->parameter.values[1]=4294967296.;
    if(mode=="clock_reordered_incarnation")std::swap(h.node->parameter.values[1],h.node->parameter.values[2]);
    if(mode=="clock_max_incarnation")h._gripper_evidence.incarnation[0]=UINT32_MAX;
    if(mode=="clock_paused_binding")h.node->parameter.values[8]-=1.;
    if(mode=="clock_invalid_calendar")r.initial_hook=[&]{r.state.robotTime.mouth=13;};
    if(mode=="clock_missing_binding")h.node->parameter.values.clear();
    if(mode=="clock_invalid_number")h.node->parameter.values[6]=std::numeric_limits<double>::quiet_NaN();
  }
  auto stop=[&]{std::lock_guard<std::mutex> lock(h._gripper_mutex);h._stop_gripper_thread=true;h._gripper_cv.notify_all();};
  if(mode=="stop_before_move")r.initial_hook=stop;
  if(mode=="error_before_move")r.initial_hook=[&]{udp_command_error=12;};
  if(mode=="stop_during_move")r.move_hook=stop;
  if(mode=="stop_before_resume")r.poll_hook=stop;
  if(mode=="clock_stop_before_resume")r.poll_hook=stop;
  if(mode=="clock_error_before_resume")r.poll_hook=[&]{udp_command_error=12;};
  if(mode=="stop_during_resume")r.resume_hook=stop;
  if(mode=="clock_stop_during_resume")r.resume_hook=stop;
  if(mode=="resume_error")r.resume_error=13;
  if(mode=="move_error")r.move_error=14;
  if(mode=="clock_move_error")r.move_error=14;
  if(mode=="clock_resume_error")r.resume_error=13;
  if(mode=="settled")r.settled=true;
  if(mode=="cached")r.stale=true;
  if(mode=="superseded" || mode=="clock_superseded")r.poll_hook=[&]{std::lock_guard<std::mutex> lock(h._gripper_mutex);
    h._pending_gripper_position=50;h._gripper_command_generation=2;
    h._last_gripper_command=.0105;h._jnt_position_command[6]=.0105;};
  std::atomic<bool> returned=false;
  std::thread worker([&]{h.gripper_worker();returned=true;});
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(!returned && h._arm_stream_paused && !h._gripper_error && std::chrono::steady_clock::now()<deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  bool finished=returned || !h._arm_stream_paused || h._gripper_error!=0;
  std::array<double,GripperExecutionEvidence::names.size()> native_values{};
  bool released=false;
  if(guarded && !h._arm_stream_paused && !h._gripper_error){
    // Completion after read is not permission to stream against the old snapshot.
    assert(h.write({}, {})==hardware_interface::return_type::OK && r.arm_sends==0);
    h.sample_gripper_evidence();native_values=h._gripper_evidence_values;
    if(mode=="clock_stale_read")h._gripper_evidence_values[10]-=1.;
    if(mode=="clock_wrong_incarnation_read")++h._gripper_evidence.incarnation[0];
    if(mode=="clock_limit")h._jnt_position_command[6]=.022;
    const auto result=h.write({}, {});
    released=result==hardware_interface::return_type::OK && r.arm_sends==1;
    assert(released==(mode=="clock_fresh" || mode=="clock_delayed_fresh" || mode=="clock_max_incarnation" || mode=="clock_superseded"));
    if(released)for(size_t i=0;i<6;++i)assert(std::abs(r.last_arm.jPos[i]*M_PI/180.-h._jnt_position_command[i])<1e-12);
    if(mode=="clock_limit")assert(h._gripper_command_generation==1 && h._last_gripper_command==.01176 && r.arm_sends==0);
  }
  stop();worker.join();assert(finished);
  bool success=mode=="completed" || mode=="settled" || mode=="superseded" ||
    mode=="clock_fresh" || mode=="clock_delayed_fresh" || mode=="clock_max_incarnation" || mode=="clock_superseded" || mode=="clock_stale_read" || mode=="clock_wrong_incarnation_read" || mode=="clock_limit";
  assert((h._gripper_evidence.completed!=0)==success);
  if(mode=="stop_before_move" || mode=="error_before_move")assert(r.moves==0 && r.resumes==0);
  if(mode=="stop_during_move" || mode=="stop_before_resume")assert(r.resumes==0);
  if(mode=="stop_during_resume")assert(r.resumes==1 && h._arm_stream_paused);
  if(mode=="superseded")assert(r.moves==2 && r.resumes==1 && h._gripper_evidence.completed==2);
  if(mode=="settled")assert(h._gripper_evidence.reason==2);
  if(guarded){
    if(mode=="clock_fresh" || mode=="clock_delayed_fresh" || mode=="clock_max_incarnation" || mode=="clock_stale_read" || mode=="clock_wrong_incarnation_read" || mode=="clock_limit")assert(r.moves==1 && r.resumes==1);
    else if(mode=="clock_superseded")assert(r.moves==2 && r.resumes==1 && h._gripper_evidence.completed==2);
    else if(mode=="clock_resume_error" || mode=="clock_late_resume" || mode=="clock_stop_during_resume")assert(r.resumes==1 && h._arm_stream_paused);
    else assert(r.resumes==0 && h._arm_stream_paused);
    if(mode=="clock_wrong_incarnation" || mode=="clock_fractional_incarnation" || mode=="clock_oversized_incarnation" ||
       mode=="clock_reordered_incarnation" || mode=="clock_paused_binding" || mode=="clock_invalid_calendar" ||
       mode=="clock_missing_binding" || mode=="clock_invalid_number")assert(r.moves==0);
  }
  // Actual native sampler method, after the thread is joined, retains stop evidence.
  h.sample_gripper_evidence();assert(h._gripper_evidence_values[25]==1.);
  auto old=h._gripper_evidence.incarnation;
  auto values=h._gripper_evidence.snapshot(h._gripper_command_generation,.01176,10.2,20.2,r.state,.021,false,false,!success,false,h._gripper_error,true);
  if(guarded && success)std::copy_n(native_values.begin(),35,values.begin());
  std::cout<<std::setprecision(17)<<"{\"moves\":"<<r.moves<<",\"resumes\":"<<r.resumes<<",\"arm_sends\":"<<r.arm_sends<<",\"released\":"<<(released?"true":"false")<<",\"wire\":[";
  for(size_t i=0;i<values.size();++i){if(i)std::cout<<",";std::cout<<values[i];}std::cout<<"],\"names\":[";
  for(size_t i=0;i<35;++i){if(i)std::cout<<",";std::cout<<"\""<<GripperExecutionEvidence::names[i]<<"\"";}std::cout<<"]}\n";
  h._gripper_evidence.activate();assert(h._gripper_evidence.incarnation!=old && h._gripper_evidence.completed==0);
}
