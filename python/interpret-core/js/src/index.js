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
      // Get up to 2 columns for display
      const columns = selector.columns.slice(0, 2);
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
  // console.log("Building options");
  const options = buildOptions(props.explanations.selector);
  const select = (
    <Select
      onChange={handleChange}
      options={options}
      defaultValue={options[props.defaultSelectValue + 1]}
    />
  );

  let renderable = <div className={"empty-space"} />;
  let name = "";

  // console.log("Selecting plot");

  if (selectedOption !== null) {
    name = props.explanations.name;

    let figure = null;
    let type = null;
    if (selectedOption === -1) {
      const overall = props.explanations.overall;
      figure = overall.figure;
      type = overall.type;
    } else {
      const specific = props.explanations.specific[selectedOption];
      figure = specific.figure;
      type = specific.type;
    }
    // console.log(type);
    // console.log(figure);

    if (type === "none") {
      renderable = (
        <div className="center-no-graph">
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
    } else if (type === "html") {
      renderable = (
        <iframe
          src={figure}
          referrerPolicy="no-referrer"
          sandbox="allow-same-origin allow-scripts"
          className="renderable-frame"
        />
      );
    } else {
      console.log(`Type ${type} not renderable.`);
    }
  }

  return (
    <div className="root">
      <div className="card">
        <div className="card-header">
          <div className="card-title">Select Component to Graph</div>
        </div>
        <div className="card-body">{select}</div>
      </div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">{name}</div>
        </div>
        <div className="card-body card-renderable">{renderable}</div>
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
