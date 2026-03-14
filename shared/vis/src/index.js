/*
Copyright (c) 2023 The InterpretML Contributors
Distributed under the MIT software license
*/

import Plotly from "plotly.js-cartesian-dist-min";
import cytoscape from "cytoscape";
import "./styles.scss";

const buildOptions = (selector) => {
  const options = [];

  // Create overall option
  options.push({ value: -1, label: "Summary" });

  // Create figure options
  for (let i = 0; i < selector.data.length; i++) {
    const columns = selector.columns.slice(0, 3);
    const record = columns
      .map((c) => `${c} (${selector.data[i][c]})`)
      .join(" | ");
    options.push({ value: i, label: `${i} : ${record}` });
  }

  return options;
};

const renderContent = (container, helpContainer, titleEl, explanations, selectedOption) => {
  container.innerHTML = "";
  helpContainer.innerHTML = "";

  if (selectedOption === null) {
    titleEl.textContent = "";
    const emptyDiv = document.createElement("div");
    emptyDiv.className = "iml-empty-space";
    container.appendChild(emptyDiv);
    return;
  }

  titleEl.textContent = explanations.name;

  let figure = null;
  let type = null;
  let help = null;
  if (selectedOption === -1) {
    const overall = explanations.overall;
    figure = overall.figure;
    type = overall.type;
    help = overall.help;
  } else {
    const specific = explanations.specific[selectedOption];
    figure = specific.figure;
    type = specific.type;
    help = specific.help;
  }

  if (type === "none") {
    const noGraphDiv = document.createElement("div");
    noGraphDiv.className = "iml-center-no-graph";
    const h1 = document.createElement("h1");
    h1.textContent = "No Overall Graph";
    noGraphDiv.appendChild(h1);
    container.appendChild(noGraphDiv);
  } else if (type === "plotly") {
    const data = figure.data;
    const layout = JSON.parse(JSON.stringify(figure.layout));
    layout.autosize = true;

    const plotDiv = document.createElement("div");
    plotDiv.style.width = "100%";
    plotDiv.style.height = "100%";
    container.appendChild(plotDiv);
    Plotly.newPlot(plotDiv, data, layout, { responsive: true });

    if (help && Object.keys(help).length > 0) {
      let helpText = help.text.trim();
      const helpDiv = document.createElement("div");
      helpDiv.className = "iml-card-help";
      if (help.link) {
        helpText = helpText + " ";
        helpDiv.appendChild(document.createTextNode(helpText));
        const a = document.createElement("a");
        a.href = help.link;
        a.textContent = "Learn more";
        helpDiv.appendChild(a);
      } else {
        helpDiv.textContent = helpText;
      }
      helpContainer.appendChild(helpDiv);
    }
  } else if (type === "html") {
    const iframe = document.createElement("iframe");
    iframe.src = figure;
    iframe.referrerPolicy = "no-referrer";
    iframe.sandbox = "allow-same-origin allow-scripts";
    iframe.className = "iml-renderable-frame";
    container.appendChild(iframe);
  } else if (type === "cytoscape") {
    const figureJson = JSON.parse(figure);
    const cyDiv = document.createElement("div");
    cyDiv.style.width = "100%";
    cyDiv.style.height = "100%";
    container.appendChild(cyDiv);
    cytoscape({
      container: cyDiv,
      elements: figureJson.elements,
      style: figureJson.stylesheet,
      layout: figureJson.layout,
    });
  } else {
    console.log(`Type ${type} not renderable.`);
  }
};

const RenderApp = (elementId, explanations, defaultSelectValue = -1) => {
  const mountNode = document.getElementById(elementId);
  mountNode.innerHTML = "";

  // Build root structure
  const root = document.createElement("div");
  root.className = "iml-root";

  // Selector card
  const selectorCard = document.createElement("div");
  selectorCard.className = "iml-card";

  const selectorHeader = document.createElement("div");
  selectorHeader.className = "iml-card-header";
  const selectorTitle = document.createElement("div");
  selectorTitle.className = "iml-card-title";
  selectorTitle.textContent = "Select Component to Graph";
  selectorHeader.appendChild(selectorTitle);
  selectorCard.appendChild(selectorHeader);

  const selectorBody = document.createElement("div");
  selectorBody.className = "iml-card-body";

  const selectEl = document.createElement("select");
  selectEl.style.width = "100%";
  selectEl.style.padding = "8px";
  selectEl.style.fontSize = "inherit";
  const options = buildOptions(explanations.selector);
  options.forEach((opt) => {
    const optionEl = document.createElement("option");
    optionEl.value = opt.value;
    optionEl.textContent = opt.label;
    if (opt.value === defaultSelectValue) {
      optionEl.selected = true;
    }
    selectEl.appendChild(optionEl);
  });
  selectorBody.appendChild(selectEl);
  selectorCard.appendChild(selectorBody);
  root.appendChild(selectorCard);

  // Renderable card
  const renderCard = document.createElement("div");
  renderCard.className = "iml-card";

  const renderHeader = document.createElement("div");
  renderHeader.className = "iml-card-header";
  const renderTitle = document.createElement("div");
  renderTitle.className = "iml-card-title";
  renderHeader.appendChild(renderTitle);
  renderCard.appendChild(renderHeader);

  const renderBody = document.createElement("div");
  renderBody.className = "iml-card-body iml-card-renderable";
  renderCard.appendChild(renderBody);

  const helpContainer = document.createElement("div");
  renderCard.appendChild(helpContainer);

  root.appendChild(renderCard);
  mountNode.appendChild(root);

  // Render initial content
  renderContent(renderBody, helpContainer, renderTitle, explanations, defaultSelectValue);

  // Handle select changes
  selectEl.addEventListener("change", (e) => {
    const value = parseInt(e.target.value, 10);
    renderContent(renderBody, helpContainer, renderTitle, explanations, value);
  });
};

export { RenderApp };
