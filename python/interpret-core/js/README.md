# interpret-inline

The interpret-inline JS library is designed to enable UX across all Jupyter notebook environments including local and cloud.

## Building and running on localhost

First install dependencies:

```sh
npm install
```

To run in hot module reloading mode:

```sh
npm start
```

To create a production build:

```sh
npm run build-prod
```

To create a development build:

```sh
npm run build-dev
```

## Using in python environment

```sh
  cp ../../../python/interpret-core/js/dist/bundle.js ../../../python/interpret-core/interpret/lib/interpret-inline.js
```

## Running

Open the file `dist/index.html` in your browser

## Credits

Made with [createapp.dev](https://createapp.dev/)
