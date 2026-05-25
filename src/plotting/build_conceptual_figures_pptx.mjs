import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const PptxGenJS = require("pptxgenjs");

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const outPath = path.join(root, "results", "figures", "conceptual_figures_fusion.pptx");
fs.mkdirSync(path.dirname(outPath), { recursive: true });

const pptx = new PptxGenJS();
pptx.author = "anonymous";
pptx.subject = "Conceptual benchmark figures";
pptx.title = "Public-signal fusion conceptual figures";
pptx.company = "anonymous";
pptx.lang = "en-US";
pptx.defineLayout({ name: "FIGURE", width: 7.1, height: 3.45 });
pptx.layout = "FIGURE";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "en-US",
};

const COLORS = {
  blue: "175A93",
  orange: "B45309",
  purple: "5B3F8C",
  green: "1F7A45",
  gray: "374151",
  rule: "111827",
  text: "111827",
  pale: "F7FAFC",
  dash: "111827",
};

function addBox(slide, { x, y, w, h, title, subtitle, color }) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: "FFFFFF", transparency: 0 },
    line: { color: COLORS.text, width: 1.85 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: x + 0.15,
    y: y + 0.10,
    w: w - 0.30,
    h: 0,
    line: { color, width: 2.0 },
  });
  slide.addText(title, {
    x,
    y: y + h * 0.22,
    w,
    h: 0.18,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: h >= 0.60 ? 15.0 : 12.2,
    bold: true,
    color: COLORS.text,
    breakLine: false,
    fit: "shrink",
  });
  slide.addText(subtitle, {
    x,
    y: y + h * 0.62,
    w,
    h: 0.13,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: h >= 0.60 ? 9.4 : 8.4,
    color: COLORS.text,
    breakLine: false,
    fit: "shrink",
  });
}

function addArrow(slide, x1, y1, x2, y2, width = 1.8) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color: COLORS.rule,
      width,
      beginArrowType: "none",
      endArrowType: "triangle",
    },
  });
}

function addSection(slide, x, y, w, h, label) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.04,
    fill: { color: "FFFFFF", transparency: 0 },
    line: { color: COLORS.text, width: 1.55, dashType: "dash" },
  });
  const labelW = Math.min(2.0, w * 0.62);
  const labelX = x + (w - labelW) / 2;
  const labelY = y + 0.03;
  slide.addShape(pptx.ShapeType.roundRect, {
    x: labelX,
    y: labelY,
    w: labelW,
    h: 0.18,
    rectRadius: 0.03,
    fill: { color: "FFFFFF", transparency: 0 },
    line: { color: COLORS.text, width: 1.1 },
  });
  slide.addText(label, {
    x: labelX,
    y: labelY + 0.035,
    w: labelW,
    h: 0.10,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: 8.8,
    bold: true,
    color: COLORS.text,
    fill: { color: "FFFFFF", transparency: 0 },
    breakLine: false,
  });
}

function addFigure1() {
  const slide = pptx.addSlide();
  slide.background = { color: "FFFFFF" };

  addSection(slide, 0.22, 0.16, 6.66, 1.02, "Input signals");

  const boxes = [
    { title: "Weather", subtitle: "Local capacity", color: COLORS.blue },
    { title: "Advisory", subtitle: "GDP/GS action", color: COLORS.orange },
    { title: "Demand", subtitle: "Schedule pressure", color: COLORS.purple },
    { title: "Outcome", subtitle: "Delay closure", color: COLORS.green },
  ];
  const topY = 0.43;
  const boxW = 1.22;
  const boxH = 0.68;
  const xs = [0.32, 1.98, 3.64, 5.30];
  boxes.forEach((b, i) => addBox(slide, { x: xs[i], y: topY, w: boxW, h: boxH, ...b }));
  addArrow(slide, xs[0] + boxW + 0.04, topY + boxH / 2, xs[1] - 0.04, topY + boxH / 2);
  addArrow(slide, xs[1] + boxW + 0.04, topY + boxH / 2, xs[2] - 0.04, topY + boxH / 2);
  addArrow(slide, xs[2] + boxW + 0.04, topY + boxH / 2, xs[3] - 0.04, topY + boxH / 2);

  addArrow(slide, 3.55, 1.18, 3.55, 1.37, 2.0);

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.43,
    y: 1.40,
    w: 6.24,
    h: 0.78,
    rectRadius: 0.06,
    fill: { color: COLORS.pale, transparency: 0 },
    line: { color: COLORS.text, width: 1.6 },
  });
  slide.addText("Residual operational state", {
    x: 0.43,
    y: 1.54,
    w: 6.24,
    h: 0.23,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: 18.0,
    bold: true,
    color: COLORS.text,
    breakLine: false,
    fit: "shrink",
  });
  slide.addText("mild weather | advisory | demand residual | outcome closure", {
    x: 0.74,
    y: 1.83,
    w: 5.62,
    h: 0.16,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: 10.4,
    color: COLORS.gray,
    breakLine: false,
    fit: "shrink",
  });

  addArrow(slide, 3.55, 2.20, 3.55, 2.41, 2.0);
  addSection(slide, 0.58, 2.43, 5.94, 0.76, "Validation evidence");

  const evidence = [
    ["active", COLORS.blue],
    ["post 3 h", COLORS.orange],
    ["matched", COLORS.purple],
    ["external", COLORS.green],
  ];
  let x = 0.82;
  for (const [label, color] of evidence) {
    slide.addShape(pptx.ShapeType.roundRect, {
      x,
      y: 2.70,
      w: 1.24,
      h: 0.34,
      rectRadius: 0.05,
      fill: { color: "FFFFFF", transparency: 0 },
      line: { color: COLORS.text, width: 1.45 },
    });
    slide.addShape(pptx.ShapeType.line, {
      x: x + 0.12,
      y: 2.755,
      w: 1.00,
      h: 0,
      line: { color, width: 1.6 },
    });
    slide.addText(label, {
      x,
      y: 2.82,
      w: 1.24,
      h: 0.12,
      margin: 0,
      align: "center",
      fontFace: "Arial",
      fontSize: 10.1,
      bold: true,
      color: COLORS.text,
      breakLine: false,
      fit: "shrink",
    });
    x += 1.42;
  }
}

function addFigure2() {
  const slide = pptx.addSlide();
  slide.background = { color: "FFFFFF" };
  addSection(slide, 0.30, 0.18, 6.50, 1.18, "Temporal windows");

  const x0 = 0.56;
  const y = 0.98;
  const timelineW = 5.98;
  slide.addShape(pptx.ShapeType.line, {
    x: x0,
    y,
    w: timelineW,
    h: 0,
    line: { color: COLORS.text, width: 1.65 },
  });
  const tickLabels = ["t-1", "t", "t+1", "t+2", "t+3"];
  tickLabels.forEach((label, i) => {
    const x = x0 + (timelineW * i) / 4;
    slide.addShape(pptx.ShapeType.line, {
      x,
      y: y - 0.055,
      w: 0,
      h: 0.11,
      line: { color: COLORS.text, width: i === 2 ? 1.55 : 1.25 },
    });
    slide.addText(label, {
      x: x - 0.22,
      y: y + 0.17,
      w: 0.44,
      h: 0.14,
      margin: 0,
      align: "center",
      fontFace: "Arial",
      fontSize: 9.4,
      color: COLORS.text,
      breakLine: false,
    });
  });

  const barY = 0.64;
  slide.addShape(pptx.ShapeType.rect, {
    x: x0 + 0.57,
    y: barY,
    w: 0.92,
    h: 0.20,
    fill: { color: "D1D5DB" },
    line: { color: COLORS.text, width: 0.8 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x0 + 1.49,
    y: barY,
    w: 1.50,
    h: 0.20,
    fill: { color: COLORS.orange },
    line: { color: COLORS.text, width: 0.8 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x0 + 2.99,
    y: barY,
    w: 2.99,
    h: 0.20,
    fill: { color: "E7A84A" },
    line: { color: COLORS.text, width: 0.8 },
  });
  [
    ["Lead horizon", x0 + 0.47, 0.45, 1.20],
    ["Active GDP/GS overlap", x0 + 1.26, 0.45, 1.92],
    ["Three-hour post-advisory window", x0 + 2.76, 0.45, 2.92],
  ].forEach(([label, x, yy, w]) => {
    slide.addText(label, {
      x,
      y: yy,
      w,
      h: 0.13,
      margin: 0,
      fontFace: "Arial",
      fontSize: 9.0,
      bold: true,
      color: COLORS.text,
      breakLine: false,
      fit: "shrink",
    });
  });

  addSection(slide, 0.44, 1.56, 6.22, 0.78, "Aligned sources");
  addArrow(slide, 3.55, 1.37, 3.55, 1.54, 2.0);
  const rows = [
    ["BTS outcomes", "scheduled arrivals and realized delay/cancellation", COLORS.green, 0.46, 2.08],
    ["IEM ASOS weather", "visibility, wind, ceiling, temperature", COLORS.blue, 2.58, 1.76],
    ["FAA ATCSCC", "GDP/GS interval overlap", COLORS.orange, 4.68, 1.60],
  ];
  for (const [title, subtitle, color, x, textW] of rows) {
    slide.addShape(pptx.ShapeType.ellipse, {
      x,
      y: 1.94,
      w: 0.12,
      h: 0.12,
      fill: { color },
      line: { color, transparency: 100 },
    });
    slide.addText(title, {
      x: x + 0.18,
      y: 1.84,
      w: textW,
      h: 0.15,
      margin: 0,
      fontFace: "Arial",
      fontSize: 9.8,
      bold: true,
      color: COLORS.text,
      breakLine: false,
      fit: "shrink",
    });
    slide.addText(subtitle, {
      x: x + 0.18,
      y: 2.07,
      w: textW,
      h: 0.16,
      margin: 0,
      fontFace: "Arial",
      fontSize: 9.8,
      color: COLORS.text,
      breakLine: false,
      fit: "shrink",
    });
  }

  addSection(slide, 0.72, 2.56, 5.66, 0.66, "Strong-advisory rule");
  addArrow(slide, 3.55, 2.36, 3.55, 2.54, 2.0);
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.93,
    y: 2.84,
    w: 5.24,
    h: 0.29,
    rectRadius: 0.05,
    fill: { color: "FFFFFF" },
    line: { color: COLORS.text, width: 1.55 },
  });
  slide.addText("at least 45 min of GDP or GS overlap", {
    x: 0.93,
    y: 2.925,
    w: 5.24,
    h: 0.12,
    margin: 0,
    align: "center",
    fontFace: "Arial",
    fontSize: 10.8,
    bold: true,
    color: COLORS.text,
    breakLine: false,
    fit: "shrink",
  });
}

addFigure1();
addFigure2();

await pptx.writeFile({ fileName: outPath });
console.log(`Wrote ${outPath}`);
