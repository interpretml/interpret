/*
Copyright (c) 2023 The InterpretML Contributors
Distributed under the MIT software license
*/

const webpack = require("webpack");
const path = require("path");

const config = {
  entry: "./src/index.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "interpret-inline.js",
    library: "interpret-inline",
    libraryTarget: "umd",
    umdNamedDefine: true
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        use: "babel-loader",
        exclude: /node_modules/
      },
      {
        test: /\.scss$/,
        use: ["style-loader", "css-loader", "sass-loader"]
      }
    ]
  },
  resolve: {
    extensions: [".js"]
  },
  devServer: {
    static: "./dist"
  }
};

module.exports = config;
