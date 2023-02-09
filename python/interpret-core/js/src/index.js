/*
Copyright (c) 2019 Microsoft Corporation
Distributed under the MIT software license
*/
/* eslint-disable react/prop-types */

import * as React from "react";
import * as ReactDOM from "react-dom";
import Plot from "react-plotly.js";
import Select from "react-select";
import { useState } from "react";
import CytoscapeComponent from 'react-cytoscapejs';
import "./styles.scss";

const App = props => {
  const [selectedOption, useSelectedOption] = useState(
    props.defaultSelectValue
  );

  const handleChange = selectedOption => {
    useSelectedOption(selectedOption.value);
  };

  const buildOptions = selector => {
    const options = [];

    // Create overall option
    const overall_option = {
      value: -1,
      label: "Summary"
    };
    options.push(overall_option);

    // Create figure options
    for (let i = 0; i < selector.data.length; i++) {
      // Get up to 3 columns for display
      const columns = selector.columns.slice(0, 3);
      const record = columns
        .map(c => `${c} (${selector.data[i][c]})`)
        .join(" | ");
      const label = `${i} : ${record}`;

      const option = {
        value: i,
        label: label
      };
      options.push(option);
    }

    return options;
  };
  const options = buildOptions(props.explanations.selector);
  const select = (
    <Select
      onChange={handleChange}
      options={options}
      defaultValue={options[props.defaultSelectValue + 1]}
    />
  );

  let renderable = <div className={"iml-empty-space"} />;
  let help_div = null;
  let name = "";

  if (selectedOption !== null) {
    name = props.explanations.name;

    let figure = null;
    let type = null;
    let help = null;
    if (selectedOption === -1) {
      const overall = props.explanations.overall;
      figure = overall.figure;
      type = overall.type;
      help = overall.help;
    } else {
      const specific = props.explanations.specific[selectedOption];
      figure = specific.figure;
      type = specific.type;
      help = specific.help;
    }

    if (type === "none") {
      renderable = (
        <div className="iml-center-no-graph">
          <h1>No Overall Graph</h1>
        </div>
      );
    } else if (type === "plotly") {
      const data = figure.data;
      const layout = JSON.parse(JSON.stringify(figure.layout));
      layout.autosize = true;
      const style = { width: "100%", height: "100%" };
      renderable = (
        <Plot
          data={data}
          layout={layout}
          style={style}
          useResizeHandler={true}
        />
      );
      
      if (help && Object.keys(help).length > 0) {
        let help_link = null;
        let help_text = help.text.trim();
        if (help.link) {
          help_text = help_text + ' '
          help_link = <a href={help.link}>Learn more</a>
        }
        help_div = (
          <div className={"iml-card-help"}>
            {help_text}{help_link}
          </div>
        );
      }
    } else if (type === "html") {
      renderable = (
        <iframe
          src={figure}
          referrerPolicy="no-referrer"
          sandbox="allow-same-origin allow-scripts"
          className="iml-renderable-frame"
        />
      );
    } else if (type === "cytoscape") {
      const figureJson = JSON.parse(figure);
      renderable = (
        <CytoscapeComponent
          elements={figureJson.elements}
          style={figureJson.style}
          stylesheet={figureJson.stylesheet}
          layout={figureJson.layout}
        />
      );
    } else {
      console.log(`Type ${type} not renderable.`);
    }
  }

  return (
    <div className="iml-root">
      <div className="iml-card">
        <div className="iml-card-header">
          <div className="iml-card-title">Select Component to Graph</div>
        </div>
        <div className="iml-card-body">{select}</div>
      </div>
      <div className="iml-card">
        <div className="iml-card-header">
          <div className="iml-card-title">{name}</div>
        </div>
        <div className="iml-card-body iml-card-renderable">{renderable}</div>
        {help_div}
      </div>
    </div>
  );
};

const RenderApp = (elementId, explanations, defaultSelectValue = -1) => {
  const mountNode = document.getElementById(elementId);
  ReactDOM.render(
    <App explanations={explanations} defaultSelectValue={defaultSelectValue} />,
    mountNode
  );
};

export { App, RenderApp };
