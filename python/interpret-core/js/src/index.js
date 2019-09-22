import React from "react";
import ReactDOM from "react-dom";
import "./styles.scss";

class App extends React.Component {
  render() {
    return <div>Hello {this.props.name}</div>;
  }
}

var mountNode = document.getElementById("app");
ReactDOM.render(<App name="Jane" />, mountNode);
