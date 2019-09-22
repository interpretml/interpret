/*
Copyright (c) 2019 Microsoft Corporation
Distributed under the MIT software license
*/
/* eslint-disable react/prop-types */

import React from "react";
import ReactDOM from "react-dom";
import "./styles.scss";

const App = props => {
  return <>Hello {props.name}</>;
};

const mountNode = document.getElementById("app");
ReactDOM.render(<App name="Jane" />, mountNode);
