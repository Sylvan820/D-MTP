(() => {
  let language = 'en';
  function refreshProgress() {
    const total = document.documentElement.scrollHeight - innerHeight;
    document.querySelector('.reading-progress span').style.width = `${total > 0 ? Math.min(100,100 * scrollY / total) : 0}%`;
    const headings = [...document.querySelectorAll(`#article-${language} h2`)];
    const current = headings.filter(h=>h.getBoundingClientRect().top < 160).at(-1);
    document.querySelectorAll('.contents a').forEach(a=>a.classList.toggle('active',current && a.hash===`#${current.id}`));
  }
  function setLanguage(next, preservePosition=false) {
    const old=document.getElementById(`article-${language}`);
    const fraction=Math.max(0,scrollY-old.offsetTop)/Math.max(1,old.offsetHeight-innerHeight);
    language=next;
    document.documentElement.lang=next==='zh'?'zh-CN':'en';
    document.querySelectorAll('[data-lang]').forEach(e=>e.hidden=e.dataset.lang!==next);
    document.querySelectorAll('article').forEach(e=>e.hidden=e.id!==`article-${next}`);
    document.querySelectorAll('[data-download]').forEach(e=>e.hidden=e.dataset.download!==next);
    document.querySelectorAll('[data-set-lang]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.setLang===next)));
    if(preservePosition && scrollY>180) {
      const active=document.getElementById(`article-${next}`);
      window.scrollTo({top:active.offsetTop+fraction*Math.max(1,active.offsetHeight-innerHeight),behavior:'instant'});
    }
    refreshProgress();
  }
  document.querySelectorAll('[data-set-lang]').forEach(b=>b.addEventListener('click',()=>setLanguage(b.dataset.setLang,true)));
  addEventListener('scroll',refreshProgress,{passive:true});
  addEventListener('resize',refreshProgress);
  setLanguage(language);
})();
