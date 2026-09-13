(function () {
'use strict';
// Editorial sequence only. Content, metrics and source evidence remain in their pages.
const story = [
  ['index.html', 'introduction', 'Robot Skill Adaptation'],
  ['index.html', 'whole', 'Closed-Loop Data Engine'],
  ['collection.html', 'condition-space', '수집 조건 설계'],
  ['collection.html', 'workspace-fit', '물체 외곽과 작업영역'],
  ['collection.html', 'approach', '정렬 동작의 시연 생성'],
  ['collection.html', 'recording-scope', '연속 시연 수집'],
  ['collection.html', 'recording-phases', '작업별 기록 범위'],
  ['data.html', 'alignment', '영상·상태·동작 정렬'],
  ['data.html', 'selection', '학습 데이터 선별'],
  ['learning.html', 'rhythm40', 'Pick & Place 학습'],
  ['learning.html', 'inputs', 'Split & Normalization'],
  ['learning.html', 'model', 'SmolVLA 동작 생성'],
  ['learning.html', 'rhythm40-comparison', '정책 비교'],
  ['architecture.html', 'native-runtime', '정책과 실물 실행'],
  ['architecture.html', 'execution-safety', 'Scene & Execution'],
  ['acquisition.html', 'policy', '실행 근거와 데이터 보완'],
  ['learning.html', 'cohort', '평가 대상의 보존'],
  ['acquisition.html', 'study', '폐루프 실험의 다음 단계'],
];
const page = location.pathname.split('/').pop();
const query = new URLSearchParams(location.search);
const mode = query.get('viewing');
let anchor;
try { anchor = decodeURIComponent(location.hash.slice(1)); } catch { anchor = ''; }
const index = story.findIndex(row => row[0] === page && row[1] === anchor);
const selected = index >= 0 ? index : !anchor ? story.findIndex(row => row[0] === page) : -1;
const main = document.querySelector('main');
const sourcePage = document.querySelector('.source-page');
function withMode(href, value) {
  const url = new URL(href, location.href);
  url.searchParams.set('viewing', value);
  return url.pathname.split('/').pop() + url.search + url.hash;
}
function storyHref(row) { return `${row[0]}?viewing=present#${row[1]}`; }

if (sourcePage && mode) {
  // Keep the authored section links; the main return goes to the screen that opened the evidence.
  const returnRow = story.find(row => row[0] === query.get('return_page') && row[1] === query.get('return_screen'));
  const primaryReturn = sourcePage.querySelector(':scope > a[href]');
  function restoreReadingReturn() {
    if (!returnRow || !primaryReturn) return;
    const target = new URL('../' + storyHref(returnRow), location.href);
    if (returnRow[1] === 'introduction' && ['pick', 'pick-place'].includes(query.get('task')))
      target.searchParams.set('task', query.get('task'));
    primaryReturn.setAttribute('href', '../' + returnRow[0] + target.search + target.hash);
    primaryReturn.textContent = '← ' + returnRow[2] + '로 돌아가기';
  }
  restoreReadingReturn();
  addEventListener('hashchange', restoreReadingReturn);
  for (const a of sourcePage.querySelectorAll('a[href]')) {
    const url = new URL(a.getAttribute('href'), location.href);
    if (url.origin === location.origin && story.some(row => url.pathname.endsWith('/' + row[0]))) {
      url.searchParams.set('viewing', mode);
      a.setAttribute('href', '../' + url.pathname.split('/').pop() + url.search + url.hash);
    }
  }
} else if (main && selected >= 0 && mode !== 'detail') {
  const row = story[selected];
  const screen = document.querySelector(`[data-reading-screen="${row[1]}"]`);
  if (screen) {
    document.body.classList.add('reading-present');
    const chapters = [
      ['introduction', 'Overview'],
      ['condition-space', 'Demonstration Design'],
      ['rhythm40', 'Learning Evidence'],
      ['native-runtime', 'Physical Rollout'],
      ['policy', 'Next Demonstration'],
    ].map(([anchor, title]) => ({start: story.findIndex(row => row[1] === anchor), title}));
    const active = chapters.filter(chapter => chapter.start <= selected).pop();
    const headerNav = document.querySelector('header nav');
    if (headerNav) {
      headerNav.setAttribute('aria-label', '포트폴리오의 주요 주제');
      headerNav.replaceChildren(...chapters.map(chapter => {
        const link = document.createElement('a');
        link.href = storyHref(story[chapter.start]); link.textContent = chapter.title;
        if (chapter === active) link.setAttribute('aria-current', 'page');
        return link;
      }));
    }

    for (const child of main.children) child.hidden = true;
    const stage = document.createElement('div');
    stage.className = 'reading-stage'; stage.tabIndex = -1;
    stage.setAttribute('aria-label', row[2]);
    screen.hidden = false;
    for (const nested of screen.querySelectorAll("[data-reading-screen]")) nested.hidden = true;
    stage.append(screen); main.append(stage);
    document.querySelector('.skip')?.addEventListener('click', event => {
      event.preventDefault(); stage.focus({preventScroll: true});
    });
    for (const video of document.querySelectorAll('video'))
      if (!stage.contains(video)) video.pause();

    const controls = document.createElement('nav');
    controls.className = 'reading-controls'; controls.setAttribute('aria-label', '발표 화면 이동');
    const previous = document.createElement('a'); previous.id = 'reading-prev';
    previous.href = selected > 0 ? storyHref(story[selected - 1]) : 'index.html?viewing=detail';
    previous.textContent = selected > 0 ? '← 이전' : '전체 보기';
    const position = document.createElement('span');
    position.textContent = `${selected + 1} / ${story.length} · ${row[2]}`;
    const detail = document.createElement('a'); detail.className = 'reading-detail';
    detail.href = withMode(`#${row[1]}`, 'detail'); detail.textContent = '상세 보기 ↗';
    const next = document.createElement('a'); next.id = 'reading-next';
    next.href = selected + 1 < story.length ? storyHref(story[selected + 1]) : storyHref(story[0]);
    next.textContent = selected + 1 < story.length ? `다음 · ${story[selected + 1][2]} →` : '처음으로 ↺';
    controls.append(previous, position, detail, next); main.append(controls);
    document.addEventListener('keydown', event => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.defaultPrevented) return;
      const target = event.target;
      if (target?.closest('input, textarea, select, button, video, audio, [contenteditable]')) return;
      const direction = {ArrowRight: next, PageDown: next, ArrowLeft: previous, PageUp: previous}[event.key];
      if (direction) { event.preventDefault(); direction.click(); }
    });
    document.querySelector('footer')?.setAttribute('hidden', '');
    for (const a of stage.querySelectorAll('a[href]')) {
      const raw = a.getAttribute('href');
      const url = new URL(raw, location.href);
      if (url.origin !== location.origin) continue;
      if (url.pathname.includes('/sources/')) {
        url.searchParams.set('viewing', 'present');
        url.searchParams.set('return_page', row[0]);
        url.searchParams.set('return_screen', row[1]);
        a.setAttribute('href', 'sources/' + url.pathname.split('/').pop() + url.search + url.hash);
      } else if (url.pathname.endsWith('.html')) {
        // Explicit side exploration stays a normal detail view; Next owns the main sequence.
        a.setAttribute('href', withMode(raw, 'detail'));
      }
    }
    if (mode === 'present') stage.focus({preventScroll: true});
    addEventListener('load', () => requestAnimationFrame(() => { stage.scrollTop = 0; }));
  }
} else if (main && !sourcePage) {
  // Old deep links remain usable, including nested disclosures and historical experiments.
  const target = document.getElementById(anchor);
  for (let node = target; node && node !== main; node = node.parentElement)
    if (node.tagName === 'DETAILS') node.open = true;
  const nearby = target?.closest('[data-reading-screen]');
  const returnIndex = story.findIndex(row => row[0] === page && row[1] === nearby?.dataset.readingScreen);
  const resume = document.createElement('a'); resume.className = 'reading-resume';
  resume.href = storyHref(story[returnIndex >= 0 ? returnIndex : Math.max(0, story.findIndex(row => row[0] === page))]);
  resume.textContent = '← 발표로 돌아가기'; main.prepend(resume);
}
if (main && !sourcePage && !document.body.classList.contains('reading-present')) {
  for (const a of main.querySelectorAll('a[href]')) {
    const url = new URL(a.getAttribute('href'), location.href);
    if (url.origin === location.origin && url.pathname.includes('/sources/')) {
      url.searchParams.set('viewing', 'detail');
      a.setAttribute('href', 'sources/' + url.pathname.split('/').pop() + url.search + url.hash);
    }
  }
}
// Native files update hash without reloading; the exported reader already routes in its shell.
if (!window.fr5Location && !sourcePage) addEventListener('hashchange', () => location.reload());

})();
