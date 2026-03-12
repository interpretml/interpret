## Building interpret-inline.js

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or later)
- npm (included with Node.js)

### All platforms (Windows, macOS, Linux)

```bash
cd shared/vis
npm install
npm run build-prod
```

The output file will be at `dist/interpret-inline.js`.

### Other scripts

| Command | Description |
|---|---|
| `npm run build-dev` | Development build (unminified, with source maps) |
| `npm run build-prod` | Production build (minified) |
| `npm run clean` | Delete the built `dist/interpret-inline.js` |
| `npm start` | Run webpack-dev-server with hot reloading |
