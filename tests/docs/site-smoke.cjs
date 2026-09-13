const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const base = (process.argv[2] || "http://127.0.0.1:8000").replace(/\/$/, "");
const screenshots = path.resolve("build/docs-screenshots");
const chapter = "/tutorials/02-bilingual-training-data";
const tokenizerChapter = "/tutorials/03-tokenizer-and-packing";
const transformerChapter = "/tutorials/04-small-transformer";
const finalChapters = [
  { route: "/tutorials/05-first-pretraining", title: "05 跑通", diagrams: 1, formulas: 2, search: "梯度累积怎样按", command: "pretrain_experiment.py --mode" },
  { route: "/tutorials/06-evaluating-a-model", title: "06 判断", diagrams: 2, formulas: 4, search: "Loss 为什么要按", command: "pretrain_experiment.py --mode" },
  { route: "/tutorials/07-resume-and-compare", title: "07 让实验", diagrams: 2, formulas: 0, search: "数据顺序也必须恢复", command: "pretrain_experiment.py --mode" },
  { route: "/tutorials/08-supervised-fine-tuning", title: "08 让 Base", diagrams: 2, formulas: 1, search: "assistant-only 到底屏蔽", command: "sft_experiment.py --mode" },
  { route: "/tutorials/09-direct-preference-optimization", title: "09 用偏好", diagrams: 1, formulas: 5, search: "policy 与 reference 各自", command: "dpo_experiment.py --mode" },
  { route: "/tutorials/10-verifiable-reward-grpo", title: "10 用可验证", diagrams: 1, formulas: 5, search: "组内 advantage 在比较", command: "grpo_experiment.py --mode" },
];
const expectedFigures = new Map([
  ["/tutorials/01-first-parameter-update", 4],
  ["/tutorials/02-bilingual-training-data", 2],
  ["/tutorials/03-tokenizer-and-packing", 2],
  ["/tutorials/04-small-transformer", 2],
  ["/tutorials/05-first-pretraining", 2],
  ["/tutorials/06-evaluating-a-model", 2],
  ["/tutorials/07-resume-and-compare", 2],
  ["/tutorials/08-supervised-fine-tuning", 0],
  ["/tutorials/09-direct-preference-optimization", 0],
  ["/tutorials/10-verifiable-reward-grpo", 0],
]);
const errors = [];

async function figuresLoaded(page, label) {
  const figures = page.locator(".tutorial-figure img");
  const total = await figures.count();
  for (let index = 0; index < total; index += 1) {
    const image = figures.nth(index);
    assert((await image.getAttribute("alt"))?.trim(),
      `${label}: every illustration must have alt text`);
    assert.equal(await image.getAttribute("loading"), index === 0 ? "eager" : "lazy");
    assert.equal(await image.getAttribute("decoding"), "async");
    await image.scrollIntoViewIfNeeded();
    await page.waitForFunction((position) => {
      const node = document.querySelectorAll(".tutorial-figure img")[position];
      return node && node.complete;
    }, index).catch(() => {});
    const source = await image.getAttribute("src");
    assert(await image.evaluate((node) => node.naturalWidth > 0),
      `${label}: the illustration ${source} must load`);
  }
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  return total;
}

async function ready(page, route, title) {
  await page.goto(`${base}/#${route}`, { waitUntil: "domcontentloaded" });
  await page.locator(".markdown-section h1").filter({ hasText: title }).waitFor();
  await page.waitForFunction((expected) =>
    document.querySelector(".markdown-section")?.dataset.route === expected,
  route.split("?")[0]);
  await page.locator(".search input").waitFor({ state: "attached" });
  const routePath = route.split("?")[0];
  const total = await figuresLoaded(page, title);
  if (expectedFigures.has(routePath)) {
    assert.equal(total, expectedFigures.get(routePath),
      `${title}: illustration count must match the reviewed set`);
  }
}

async function noOverflow(page) {
  assert(await page.evaluate(() =>
    document.documentElement.scrollWidth <= window.innerWidth + 1
  ), "The page must not scroll horizontally");
}

async function main() {
  fs.mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      permissions: ["clipboard-read", "clipboard-write"],
      reducedMotion: "reduce",
    });
    context.on("page", (page) => page.on("pageerror", (error) => errors.push(error.message)));
    const page = await context.newPage();
    await ready(page, "/", "LLM Lifecycle Lab");
    await page.locator(".mermaid-source svg").waitFor();
    await noOverflow(page);
    assert.equal(await page.locator(".chapter-pagination .next").count(), 1);
    assert.equal(await page.locator(
      '.sidebar a[href^="#/tutorials/0"], .sidebar a[href="#/tutorials/10-verifiable-reward-grpo"]'
    ).count(), 10);
    await page.screenshot({ path: path.join(screenshots, "desktop-home.png") });

    await page.locator('.sidebar a[href="#/tutorials/02-bilingual-training-data"]').click();
    await page.locator(".markdown-section h1").filter({ hasText: "02 准备" }).waitFor();
    await page.waitForFunction(() => document.querySelectorAll(".mermaid-source svg").length === 3);
    assert.equal(await page.locator(".katex-error").count(), 0);
    assert.equal(await page.locator(".math-block .katex").count(), 1);
    assert.equal(await page.locator(".sample-pair").count(), 1);
    assert.equal(await page.locator(".chapter-pagination a").first().getAttribute("href"),
      "#/tutorials/01-first-parameter-update");
    await noOverflow(page);
    await page.screenshot({ path: path.join(screenshots, "desktop-chapter2.png") });

    await page.getByRole("button", { name: "切换深色模式", exact: true }).click();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.locator(".math-block .katex").waitFor();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await page.screenshot({ path: path.join(screenshots, "desktop-dark.png") });
    await page.getByRole("button", { name: "切换浅色模式", exact: true }).click();

    const experiment = page.locator("details").filter({
      has: page.locator("summary", { hasText: "运行实验" }),
    });
    await experiment.locator("summary").click();
    const code = await experiment.locator("pre code").textContent();
    await experiment.getByRole("button", { name: "复制代码", exact: true }).click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), code);
    const sourceLink = await page.locator("a").filter({ hasText: /^split\.py$/ }).getAttribute("href");
    assert.equal(sourceLink,
      "https://github.com/wang-TJ-20/llm-lifecycle-lab/blob/main/src/llm_lifecycle_lab/data/split.py");

    const search = page.getByRole("searchbox", { name: "搜索教程" });
    await page.waitForFunction(() => Object.keys(localStorage).some((key) =>
      key.startsWith("docsify.search.index") && localStorage.getItem(key).includes("source_id")));
    await search.fill("source_id");
    await page.locator(".matching-post a").first().waitFor();
    assert(await page.locator(".matching-post a").count() > 0);
    await page.locator(".matching-post a").first().click();
    await page.locator(".markdown-section h1").waitFor();
    await search.fill("zzznomatchingchapterzzz");
    await page.getByText("没有找到相关内容", { exact: true }).waitFor();
    await page.getByRole("button", { name: "清除搜索" }).click();
    assert.equal(await search.inputValue(), "");

    await ready(page, `${chapter}?id=获取固定来源`, "02 准备");
    assert(await page.locator("details[open]").count() >= 1,
      "Deep links must reveal the collapsed section");

    await page.locator(".chapter-pagination .next").click();
    await page.locator(".markdown-section h1").filter({ hasText: "03 让" }).waitFor();
    await page.waitForFunction(() =>
      document.querySelectorAll(".mermaid-source svg").length === 2);
    assert.equal(await page.locator(".math-block .katex").count(), 2);
    assert.equal(await page.locator(".katex-error").count(), 0);
    assert.equal(await page.locator(".markdown-section p").filter({ hasText: "**" }).count(), 0);
    assert.equal(await page.locator(".chapter-pagination .next").getAttribute("href"),
      `#${transformerChapter}`);
    assert.equal(await page.locator(".chapter-pagination a").first().getAttribute("href"),
      `#${chapter}`);
    assert.equal(await page.locator("td code").filter({ hasText: /^<\|bos\|>$/ }).count(), 1);
    assert.equal(await page.locator("a").filter({ hasText: /^training\/logprobs\.py$/ }).getAttribute("href"),
      "https://github.com/wang-TJ-20/llm-lifecycle-lab/blob/main/src/llm_lifecycle_lab/training/logprobs.py");
    await noOverflow(page);
    await page.screenshot({ path: path.join(screenshots, "desktop-chapter3.png") });
    const tokenizerExperiment = page.locator("details").filter({
      has: page.locator("summary", { hasText: "运行词表与 Packing" }),
    });
    await tokenizerExperiment.locator("summary").click();
    const tokenizerCode = await tokenizerExperiment.locator("pre code").textContent();
    await tokenizerExperiment.getByRole("button", { name: "复制代码", exact: true }).click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), tokenizerCode);
    await search.fill("词表越大");
    await page.locator(`.matching-post a[href*="${tokenizerChapter}"]`).first().waitFor();
    await page.getByRole("button", { name: "清除搜索" }).click();

    await page.locator(".markdown-section a").filter({ hasText: /^环境准备$/ }).click();
    await page.locator(".markdown-section h1").filter({ hasText: "Pretrain 训练文档" }).waitFor();
    await page.waitForFunction(() => {
      const heading = document.getElementById("_2-环境准备");
      const top = heading?.getBoundingClientRect().top;
      return window.scrollY > 0 && top >= 0 && top <= 150;
    });
    assert.equal(new URLSearchParams(new URL(page.url()).hash.split("?")[1]).get("id"),
      "_2-环境准备");

    for (const [name, title] of [
      ["产物复用说明", "8. 产物校验、复用与迁移"],
      ["Packing 到底保存了什么", "7. Packing 到底保存了什么"],
    ]) {
      await ready(page, tokenizerChapter, "03 让");
      if (name.startsWith("Packing")) {
        await page.locator("details").filter({
          has: page.locator("summary", { hasText: "真实 Smoke" }),
        }).locator("summary").click();
      }
      await page.getByRole("link", { name, exact: true }).click();
      await page.waitForFunction((title) => {
        const heading = [...document.querySelectorAll(".markdown-section h2")]
          .find((node) => node.textContent === title);
        const top = heading?.getBoundingClientRect().top;
        return window.scrollY > 0 && top >= 0 && top < window.innerHeight;
      }, title);
    }

    await ready(page, tokenizerChapter, "03 让");
    await page.locator(".chapter-pagination .next").click();
    await page.locator(".markdown-section h1").filter({ hasText: "04 搭建" }).waitFor();
    await page.waitForFunction(() =>
      document.querySelectorAll(".mermaid-source svg").length === 3);
    assert.equal(await page.locator(".math-block .katex").count(), 7);
    assert.equal(await page.locator(".katex-error, .diagram-error").count(), 0);
    assert.equal(await page.locator(".markdown-section p").filter({ hasText: "**" }).count(), 0);
    assert.equal(await page.locator(".chapter-pagination .next").getAttribute("href"),
      `#${finalChapters[0].route}`);
    assert.equal(await page.locator(".chapter-pagination a").first().getAttribute("href"),
      `#${tokenizerChapter}`);
    assert.equal(await page.getByRole("link", { name: "transformer.py", exact: true }).getAttribute("href"),
      "https://github.com/wang-TJ-20/llm-lifecycle-lab/blob/main/src/llm_lifecycle_lab/model/native/transformer.py");
    await page.evaluate(() => window.scrollTo(0, 0));
    await noOverflow(page);
    await page.screenshot({ path: path.join(screenshots, "desktop-chapter4.png") });
    await page.getByRole("button", { name: "切换深色模式", exact: true }).click();
    await page.waitForFunction(() =>
      document.querySelectorAll(".mermaid-source svg").length === 3);
    await page.screenshot({ path: path.join(screenshots, "desktop-chapter4-dark.png") });
    await page.getByRole("button", { name: "切换浅色模式", exact: true }).click();
    for (const name of ["Attention 公式与因果遮罩", "微型 Transformer"]) {
      const experiment = page.locator("details").filter({
        has: page.locator("summary", { hasText: name }),
      });
      await experiment.locator("summary").click();
      const code = await experiment.locator("pre code").textContent();
      await experiment.getByRole("button", { name: "复制代码", exact: true }).click();
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), code);
    }
    await search.fill("SwiGLU 怎样加工");
    await page.locator(`.matching-post a[href*="${transformerChapter}"]`).first().waitFor();
    await page.getByRole("button", { name: "清除搜索" }).click();

    for (const [index, lesson] of finalChapters.entries()) {
      await ready(page, lesson.route, lesson.title);
      await page.waitForFunction((count) =>
        document.querySelectorAll(".mermaid-source svg").length === count,
      lesson.diagrams);
      assert.equal(await page.locator(".math-block .katex").count(), lesson.formulas);
      assert.equal(await page.locator(".katex-error, .diagram-error").count(), 0);
      assert(await page.locator(".markdown-section p").evaluateAll((nodes) =>
        nodes.every((node) => {
          const copy = node.cloneNode(true);
          copy.querySelectorAll("code").forEach((code) => code.remove());
          return !copy.textContent.includes("**");
        })
      ), `${lesson.title}: emphasis must render outside code spans`);
      assert.equal(await page.locator(".chapter-pagination a").first().getAttribute("href"),
        `#${index === 0 ? transformerChapter : finalChapters[index - 1].route}`);
      if (index < finalChapters.length - 1) {
        assert.equal(await page.locator(".chapter-pagination .next").getAttribute("href"),
          `#${finalChapters[index + 1].route}`);
        await page.locator(".chapter-pagination .next").click();
        await page.locator(".markdown-section h1").filter({
          hasText: finalChapters[index + 1].title,
        }).waitFor();
        await ready(page, lesson.route, lesson.title);
      } else {
        assert.equal(await page.locator(".chapter-pagination .next").count(), 0);
      }
      await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
      await noOverflow(page);
      await page.screenshot({ path: path.join(screenshots, `desktop-chapter${index + 5}.png`), animations: "disabled" });
      const command = page.locator(".markdown-section > pre").filter({
        hasText: `python scripts/${lesson.command}`,
      }).first();
      const code = await command.locator("code").textContent();
      await command.getByRole("button", { name: "复制代码", exact: true }).click();
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), code);
      await search.fill(lesson.search);
      await page.locator(`.matching-post a[href*="${lesson.route}"]`).first().waitFor();
      await page.getByRole("button", { name: "清除搜索" }).click();
    }

    await ready(page, "/NATIVE_MODEL_GUIDE", "自有模型介绍");
    await noOverflow(page);
    await ready(page, "/experiments/native-60m-baseline-v1", "60M CUDA Reference");
    await page.locator('img[src*="native-60m-baseline-v1/loss.svg"]').waitFor();
    assert.equal(
      await page.locator('img[src*="native-60m-baseline-v1/"]').count(),
      5,
    );
    await noOverflow(page);
    await page.screenshot({
      path: path.join(screenshots, "desktop-60m-baseline-candidate.png"),
      animations: "disabled",
    });
    await ready(page, "/does-not-exist", "没有找到这一页");

    const phone = await context.newPage();
    await phone.setViewportSize({ width: 390, height: 844 });
    await ready(phone, chapter, "02 准备");
    await phone.waitForFunction(() => document.querySelectorAll(".mermaid-source svg").length === 3);
    await phone.waitForFunction(() => document.querySelector(".sidebar").getBoundingClientRect().right <= 0);
    await noOverflow(phone);
    assert.equal(await phone.getByRole("button", { name: "打开课程目录" }).getAttribute("aria-expanded"), "false");
    assert.equal(await phone.locator(".site-header").evaluate((node) => node.getBoundingClientRect().top), 0);
    await phone.screenshot({ path: path.join(screenshots, "mobile-chapter2.png"), animations: "disabled" });
    await phone.getByRole("button", { name: "打开课程目录" }).click();
    await phone.waitForFunction(() => document.querySelector(".sidebar").getBoundingClientRect().left === 0);
    assert.equal(await phone.locator("#menu-toggle").getAttribute("aria-expanded"), "true");
    assert(await phone.evaluate(() => document.querySelector(".content").inert));
    await phone.screenshot({ path: path.join(screenshots, "mobile-menu.png"), animations: "disabled" });
    await phone.locator('.sidebar a[href="#/tutorials/01-first-parameter-update"]').click();
    await phone.locator(".markdown-section h1").filter({ hasText: "01 从" }).waitFor();
    assert.equal(await phone.locator("#menu-toggle").getAttribute("aria-expanded"), "false");
    await phone.getByRole("button", { name: "打开课程目录" }).click();
    await phone.keyboard.press("Escape");
    assert.equal(await phone.locator("#menu-toggle").getAttribute("aria-expanded"), "false");
    await phone.setViewportSize({ width: 320, height: 720 });
    await noOverflow(phone);
    await phone.screenshot({ path: path.join(screenshots, "mobile-320.png"), animations: "disabled" });

    for (const width of [390, 320]) {
      await phone.setViewportSize({ width, height: width === 320 ? 720 : 844 });
      await ready(phone, tokenizerChapter, "03 让");
      await phone.waitForFunction(() =>
        document.querySelectorAll(".mermaid-source svg").length === 2);
      await phone.evaluate(() => {
        document.querySelectorAll("details[open]").forEach((node) => { node.open = false; });
        window.scrollTo(0, 0);
      });
      await noOverflow(phone);
      assert(await phone.locator(".math-block").evaluateAll((nodes) =>
        nodes.every((node) => node.scrollWidth <= node.clientWidth + 1)
      ), "Chapter 3 formulas must fit on a narrow screen");
      await phone.screenshot({
        path: path.join(screenshots, `mobile-chapter3-${width}.png`),
        animations: "disabled",
      });
      const packingFormula = phone.locator(".math-block").last();
      await packingFormula.scrollIntoViewIfNeeded();
      await phone.screenshot({
        path: path.join(screenshots, `mobile-chapter3-packing-${width}.png`),
        animations: "disabled",
      });
      await phone.locator("details").filter({
        has: phone.locator("summary", { hasText: "运行词表与 Packing" }),
      }).locator("summary").click();
      await noOverflow(phone);
    }

    for (const width of [390, 320]) {
      await phone.setViewportSize({ width, height: width === 320 ? 720 : 844 });
      await ready(phone, transformerChapter, "04 搭建");
      await phone.waitForFunction(() =>
        document.querySelectorAll(".mermaid-source svg").length === 3);
      await phone.evaluate(() => window.scrollTo(0, 0));
      await noOverflow(phone);
      assert(await phone.locator(".math-block").evaluateAll((nodes) =>
        nodes.every((node) => node.scrollWidth <= node.clientWidth + 1)
      ), "Chapter 4 formulas must fit on a narrow screen");
      await phone.screenshot({
        path: path.join(screenshots, `mobile-chapter4-${width}.png`),
        animations: "disabled",
      });
      for (const [name, locator] of [
        ["residual", phone.locator(".diagram").nth(1)],
        ["gqa", phone.locator(".diagram").last()],
        ["rope", phone.locator(".math-block").nth(4)],
      ]) {
        await locator.scrollIntoViewIfNeeded();
        await phone.screenshot({
          path: path.join(screenshots, `mobile-chapter4-${name}-${width}.png`),
          animations: "disabled",
        });
      }
      await phone.locator("details").filter({
        has: phone.locator("summary", { hasText: "微型 Transformer" }),
      }).locator("summary").click();
      await noOverflow(phone);
    }

    for (const width of [390, 320]) {
      await phone.setViewportSize({ width, height: width === 320 ? 720 : 844 });
      for (const [index, lesson] of finalChapters.entries()) {
        await ready(phone, lesson.route, lesson.title);
        await phone.waitForFunction((count) =>
          document.querySelectorAll(".mermaid-source svg").length === count,
        lesson.diagrams);
        await phone.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
        await noOverflow(phone);
        assert(await phone.locator(".math-block").evaluateAll((nodes) =>
          nodes.every((node) => node.scrollWidth <= node.clientWidth + 1)
        ), `${lesson.title}: formulas must fit on mobile`);
        await phone.screenshot({
          path: path.join(screenshots, `mobile-chapter${index + 5}-${width}.png`),
          animations: "disabled",
        });
        await phone.locator(".diagram").first().scrollIntoViewIfNeeded();
        await phone.screenshot({
          path: path.join(screenshots, `mobile-chapter${index + 5}-diagram-${width}.png`),
          animations: "disabled",
        });
        if (lesson.formulas) {
          await phone.locator(".math-block").last().scrollIntoViewIfNeeded();
          await phone.screenshot({
            path: path.join(screenshots, `mobile-chapter${index + 5}-formula-${width}.png`),
            animations: "disabled",
          });
        }
        await phone.locator("details summary").first().click();
        await noOverflow(phone);
      }
    }

    await phone.setViewportSize({ width: 320, height: 720 });
    await ready(phone, "/experiments/native-60m-baseline-v1", "60M CUDA Reference");
    await phone.locator('img[src*="native-60m-baseline-v1/loss.svg"]').waitFor();
    await noOverflow(phone);
    await phone.screenshot({
      path: path.join(screenshots, "mobile-60m-baseline-candidate.png"),
      animations: "disabled",
    });

    const prefixed = await context.newPage();
    await prefixed.route(`${base}/pages-preview/**`, async (route) => {
      const response = await route.fetch({
        url: route.request().url().replace(`${base}/pages-preview/`, `${base}/`),
      });
      await route.fulfill({ response });
    });
    await prefixed.goto(`${base}/pages-preview/#${finalChapters[2].route}`, { waitUntil: "domcontentloaded" });
    await prefixed.locator(".markdown-section h1").filter({ hasText: "07 让实验" }).waitFor();
    await prefixed.locator(".mermaid-source svg").first().waitFor();
    await prefixed.locator(".chapter-links a").filter({ hasText: "上一篇" }).first().click();
    await prefixed.locator(".markdown-section h1").filter({ hasText: "06 判断" }).waitFor();
    assert(new URL(prefixed.url()).pathname.startsWith("/pages-preview/"));
    assert.equal(await figuresLoaded(prefixed, "Pages 子路径"), 2,
      "Chapter 6 must retain its reviewed illustrations under a Pages subpath");
    assert(await prefixed.locator(".tutorial-figure img").evaluateAll((nodes) =>
      nodes.every((node) => new URL(node.currentSrc || node.src).pathname.startsWith("/pages-preview/"))
    ), "Illustrations must resolve inside the Pages subpath");
    assert.deepEqual(errors, [], "No uncaught browser errors");
    console.log("PASS: tutorials and experiment record, desktop/mobile, diagrams, illustrations, formulas, theme, clipboard, search, source links, deep links, 404, Pages subpath, and overflow checks.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
