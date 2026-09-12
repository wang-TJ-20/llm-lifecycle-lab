/* The Markdown remains the source of truth; this file only supplies the reader. */
(function () {
  "use strict";

  const repository = "https://github.com/wang-TJ-20/llm-lifecycle-lab";
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
  const icon = (name) => `<i data-lucide="${name}" aria-hidden="true"></i>`;
  const icons = () => window.lucide?.createIcons();
  const menu = document.getElementById("menu-toggle");
  const backdrop = document.getElementById("menu-backdrop");
  const themeButton = document.getElementById("theme-toggle");
  let headings = [];
  let outlineLinks = [];
  let diagramQueue = Promise.resolve();
  const mobile = window.matchMedia("(max-width: 760px)");

  function syncMenuAccess() {
    const open = document.body.classList.contains("reading-menu-open");
    const sidebar = document.querySelector(".sidebar");
    if (sidebar) {
      sidebar.inert = mobile.matches && !open;
      sidebar.setAttribute("aria-hidden", String(mobile.matches && !open));
    }
    const content = document.querySelector(".content");
    if (content) content.inert = mobile.matches && open;
  }

  function closeMenu(returnFocus = false) {
    document.body.classList.remove("reading-menu-open");
    menu.setAttribute("aria-expanded", "false");
    menu.setAttribute("aria-label", "打开课程目录");
    backdrop.hidden = true;
    syncMenuAccess();
    if (returnFocus) menu.focus();
  }

  menu.addEventListener("click", () => {
    const open = document.body.classList.toggle("reading-menu-open");
    menu.setAttribute("aria-expanded", String(open));
    menu.setAttribute("aria-label", open ? "关闭课程目录" : "打开课程目录");
    backdrop.hidden = !open;
    syncMenuAccess();
    if (open) document.querySelector(".search input")?.focus();
  });
  backdrop.addEventListener("click", () => closeMenu(true));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu(true);
  });
  document.addEventListener("click", (event) => {
    if (event.target.closest(".sidebar a, .site-brand")) closeMenu();
    if (event.target.closest(".skip-link")) {
      event.preventDefault();
      const main = document.getElementById("main");
      main?.setAttribute("tabindex", "-1");
      main?.focus();
    }
  });
  mobile.addEventListener("change", () => closeMenu());

  function revealAnchor(force = false) {
    const query = location.hash.split("?")[1];
    const id = new URLSearchParams(query || "").get("id");
    if (!id) return;
    let target = document.getElementById(id);
    let alias = false;
    if (!target) {
      // GitHub and Docsify differ on punctuation in heading IDs. Never guess on collisions.
      const comparable = (value) => value.replace(/[\p{P}\p{S}]/gu, "");
      const candidates = [...document.querySelectorAll(
        ".markdown-section :is(h1, h2, h3, h4, h5, h6)[id]"
      )].filter((heading) => comparable(heading.id) === comparable(id));
      if (candidates.length === 1) {
        target = candidates[0];
        alias = true;
      }
    }
    if (!target) return;
    let parent = target.parentElement;
    let opened = false;
    while (parent) {
      if (parent.tagName === "DETAILS" && !parent.open) {
        parent.open = true;
        opened = true;
      }
      parent = parent.parentElement;
    }
    if (opened || force || alias) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
  }
  window.addEventListener("hashchange", () => {
    closeMenu();
    requestAnimationFrame(() => revealAnchor());
  });

  function syncThemeButton() {
    const dark = document.documentElement.dataset.theme === "dark";
    themeButton.innerHTML = icon(dark ? "sun" : "moon");
    themeButton.title = dark ? "切换浅色模式" : "切换深色模式";
    themeButton.setAttribute("aria-label", themeButton.title);
    themeButton.setAttribute("aria-pressed", String(dark));
    icons();
  }

  function renderDiagrams() {
    // Serialize Mermaid renders so a quick theme/route change cannot race its renderer.
    diagramQueue = diagramQueue.catch(() => {}).then(async () => {
      const nodes = [...document.querySelectorAll(".mermaid-source")];
      if (!nodes.length) return;
      const dark = document.documentElement.dataset.theme === "dark";
      if (!window.mermaid) {
        nodes.forEach((node) => node.classList.add("diagram-error"));
        return;
      }
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "base",
        fontFamily: '"PingFang SC", "Microsoft YaHei", sans-serif',
        themeVariables: {
          primaryColor: dark ? "#253d36" : "#eaf5f1",
          primaryTextColor: dark ? "#e3e8eb" : "#252b30",
          primaryBorderColor: dark ? "#75d5bc" : "#087d70",
          lineColor: dark ? "#a5afb5" : "#687178",
          secondaryColor: dark ? "#393124" : "#fcf7ea",
          tertiaryColor: dark ? "#23282c" : "#f5f7f8",
          fontSize: "15px",
        },
        flowchart: { htmlLabels: false, useMaxWidth: true, curve: "linear" },
      });
      nodes.forEach((node) => {
        node.dataset.source ||= node.textContent;
        node.textContent = node.dataset.source;
        node.removeAttribute("data-processed");
      });
      try {
        await window.mermaid.run({ nodes });
      } catch (error) {
        console.error("Diagram rendering failed", error);
        nodes.filter((node) => node.isConnected).forEach((node) => {
          node.textContent = node.dataset.source;
          node.classList.add("diagram-error");
        });
      }
      updateProgress();
    });
    return diagramQueue;
  }

  themeButton.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("llmlab-theme", theme); } catch (_) { /* Optional preference. */ }
    syncThemeButton();
    renderDiagrams();
  });
  syncThemeButton();

  function updateProgress() {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    const progress = available > 0 ? Math.min(1, Math.max(0, window.scrollY / available)) : 0;
    document.querySelector(".reading-progress").style.transform = `scaleX(${progress})`;
    let current = 0;
    headings.forEach((heading, index) => {
      if (heading.getBoundingClientRect().top <= 130) current = index;
    });
    outlineLinks.forEach((link, index) => {
      link.classList.toggle("active", index === current);
      if (index === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }
  let scrolling = false;
  window.addEventListener("scroll", () => {
    if (scrolling) return;
    scrolling = true;
    requestAnimationFrame(() => { updateProgress(); scrolling = false; });
  }, { passive: true });

  async function copyCode(button, text) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(text);
      button.innerHTML = icon("check");
      button.title = "已复制";
      document.getElementById("copy-status").textContent = "代码已复制";
      setTimeout(() => {
        button.innerHTML = icon("copy");
        button.title = "复制代码";
        icons();
      }, 1600);
    } catch (_) {
      button.title = "复制失败，请选中代码复制";
      document.getElementById("copy-status").textContent = button.title;
    }
    icons();
  }

  function pagePath(vm) {
    const path = vm.route.path;
    return path.endsWith("/") ? `${path}README.md` : path.endsWith(".md") ? path : `${path}.md`;
  }

  function readerPlugin(hook, vm) {
    hook.afterEach((html) => {
      const fragment = new DOMParser().parseFromString(html, "text/html");
      const h1 = fragment.querySelector("h1");
      const words = fragment.body.textContent.length;
      const category = vm.route.path.startsWith("/tutorials/") ? "实践系列" :
        vm.route.path === "/" ? "学习路线" : "参考指南";
      if (h1) {
        const meta = fragment.createElement("div");
        meta.className = "page-meta";
        meta.innerHTML = `<span class="category">${category}</span><span class="meta-dot"></span>` +
          `<span>约 ${Math.max(1, Math.ceil(words / 500))} 分钟</span>`;
        h1.before(meta);
      }
      fragment.querySelectorAll("p").forEach((paragraph) => {
        if (paragraph.querySelector("a") && /系列目录/.test(paragraph.textContent) &&
          paragraph.textContent.length < 100) paragraph.classList.add("chapter-links");
      });
      fragment.querySelectorAll(".tutorial-figure img").forEach((image, order) => {
        image.setAttribute("loading", order === 0 ? "eager" : "lazy");
        image.setAttribute("decoding", "async");
      });
      return fragment.body.innerHTML;
    });

    hook.doneEach(() => {
      const article = document.querySelector(".markdown-section");
      if (!article) return;
      article.dataset.route = vm.route.path;
      const title = article.querySelector("h1")?.textContent || "实践教程";
      document.title = title === "LLM Lifecycle Lab" ? `${title} · 实践教程` : `${title} · LLM Lifecycle Lab`;
      document.querySelector(".sidebar")?.setAttribute("id", "course-sidebar");
      document.querySelector(".sidebar")?.setAttribute("aria-label", "课程目录");
      syncMenuAccess();
      const search = document.querySelector(".search input");
      search?.setAttribute("aria-label", "搜索教程");
      const clear = document.querySelector(".search .clear-button");
      if (clear && !clear.dataset.accessible) {
        clear.dataset.accessible = "true";
        clear.setAttribute("role", "button");
        clear.setAttribute("tabindex", "0");
        clear.setAttribute("aria-label", "清除搜索");
        clear.title = "清除搜索";
        clear.innerHTML = icon("x");
        clear.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            clear.click();
            search?.focus();
          }
        });
      }
      article.querySelectorAll("pre").forEach((pre) => {
        const code = pre.querySelector("code");
        if (!code || pre.querySelector(".copy-code")) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "copy-code icon-button";
        button.title = "复制代码";
        button.setAttribute("aria-label", "复制代码");
        button.innerHTML = icon("copy");
        button.addEventListener("click", () => copyCode(button, code.textContent));
        pre.append(button);
      });
      article.querySelectorAll(".math-block").forEach((node) => {
        if (window.katex) window.katex.render(node.textContent, node, {
          displayMode: true, throwOnError: false, strict: "warn", trust: false,
        });
      });
      headings = [...article.querySelectorAll("h2")].filter((heading) => !heading.closest("details"));
      const outline = document.getElementById("page-outline");
      outline.innerHTML = headings.length ? `<p>本章目录</p><ul>${headings.map((heading) =>
        `<li><a href="${escapeHtml(heading.querySelector("a")?.getAttribute("href") || "#/")}">` +
        `${escapeHtml(heading.textContent)}</a></li>`).join("")}</ul>` : "";
      outlineLinks = [...outline.querySelectorAll("a")];

      const chapters = [...document.querySelectorAll(".sidebar-nav a")].filter((link) =>
        /#\/tutorials\/\d/.test(link.href));
      const index = chapters.findIndex((link) =>
        new URL(link.href).hash.split("?")[0] === `#${vm.route.path}`);
      if (index >= 0 || vm.route.path === "/") {
        const previous = chapters[index - 1];
        const next = chapters[index + 1];
        const pagination = document.createElement("nav");
        pagination.className = "chapter-pagination";
        pagination.setAttribute("aria-label", "章节翻页");
        pagination.innerHTML = previous
          ? `<a href="${previous.getAttribute("href")}">${icon("arrow-left")}<span><small>上一篇</small>${escapeHtml(previous.textContent)}</span></a>`
          : `<a href="#/tutorials/">${icon("list")}<span>系列目录</span></a>`;
        if (next) pagination.innerHTML += `<a class="next" href="${next.getAttribute("href")}"><span><small>下一篇</small>${escapeHtml(next.textContent)}</span>${icon("arrow-right")}</a>`;
        article.append(pagination);
      }
      const footer = document.createElement("footer");
      footer.className = "page-footer";
      footer.innerHTML = `<span>LLM Lifecycle Lab · 学习与实验</span>` +
        `<a href="${repository}/blob/main/docs${pagePath(vm)}" target="_blank" rel="noopener noreferrer">` +
        `${icon("file-pen-line")}查看本页源码</a>`;
      article.append(footer);
      icons();
      renderDiagrams().then(() => revealAnchor(true));
      updateProgress();
    });
  }

  window.$docsify = function (vm) {
    return {
      name: "LLM Lifecycle Lab",
      loadSidebar: true,
      alias: { "/.*/_sidebar.md": "/_sidebar.md" },
      relativePath: true,
      auto2top: true,
      subMaxLevel: 0,
      topMargin: 90,
      notFoundPage: "_404.md",
      externalLinkTarget: "_blank",
      search: {
        paths: "auto", depth: 3, maxAge: 600000,
        placeholder: "搜索教程…", noData: "没有找到相关内容",
        namespace: `llmlab-reader-v1:${location.pathname}`,
        hideOtherSidebarContent: true,
      },
      markdown: {
        renderer: {
          link(href, title, text) {
            // Preserve repository-relative Markdown links without publishing source code.
            if (href && !/^(?:[a-z]+:|\/|#)/i.test(href)) {
              const target = new URL(href, `https://repo.invalid/docs${pagePath(vm)}`);
              if (!target.pathname.startsWith("/docs/")) {
                return `<a href="${repository}/blob/main${escapeHtml(target.pathname + target.hash)}"` +
                  ` target="_blank" rel="noopener noreferrer">${text}</a>`;
              }
              // Docsify prefixes numeric heading IDs; GitHub Markdown does not.
              if (/^#\d/.test(target.hash)) href = href.replace(/#(?=\d)/, "#_");
              if (target.pathname.endsWith("/README.md")) {
                return this.origin.link.call(this, href.replace(/README\.md(?=#|$)/, ""), title, text);
              }
            }
            return this.origin.link.call(this, href, title, text);
          },
          code(code, language) {
            if (language === "mermaid") {
              return `<figure class="diagram"><div class="mermaid-source">${escapeHtml(code)}</div></figure>`;
            }
            if (language === "math") return `<div class="math-block">${escapeHtml(code)}</div>`;
            return this.origin.code.apply(this, arguments);
          },
        },
      },
      plugins: [...(window.$docsify.plugins || []), readerPlugin],
    };
  };
})();
