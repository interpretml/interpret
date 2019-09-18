// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.
//
// Some static assets may be required by the custom widget javascript. The base
// url for the notebook is not known at build time and is therefore computed
// dynamically.
(window as any).__webpack_public_path__ = document.querySelector('body')!.getAttribute('data-base-url') + 'nbextensions/interpret-ml-widget';


// Configure requirejs
if ((window as any).require) {
    (window as any).require.config({
        map: {
            "*" : {
                "interpret-ml-widget": "nbextensions/interpret-ml-widget/index",
            }
        }
    });
}

// Export the required load_ipython_extension
export * from './index';
