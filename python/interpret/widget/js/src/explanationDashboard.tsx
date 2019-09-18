import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';
import { ExplanationDashboard } from 'mlchartlib';
import * as _ from 'lodash';
import React from 'react';
import ReactDOM from 'react-dom';

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
export class  ExplanationModel extends DOMWidgetModel {
    defaults() {
        return {
            _model_name : 'ExplanationModel',
            _view_name : 'ExplanationView',
            _model_module : 'interpret-ml-widget',
            _view_module : 'interpret-ml-widget',
            _model_module_version : '0.1.0',
            _view_module_version : '0.1.0',
            value: {},
            request: {},
            response: {}
        }
    }
};

interface IPromiseResolvers {
    resolve: (value: any) => void;
    reject: (error: any) => void;
    timeout: number;
}

// Custom View. Renders the widget model.
export class ExplanationView extends DOMWidgetView {
    el: any;
    private requestIndex: number = 0;
    private promiseDict: {[key: number]: IPromiseResolvers} = {};
    private messages = {
        'LocalExpAndTestReq': [{displayText: 'This view requires local explanations'}],
        'LocalOrGlobalAndTestReq': [{displayText: 'This view requires either local or global explanations'}],
        'TestReq': [{displayText: 'This view requires a dataset to visualize'}],
        'PredictorReq': [{displayText: 'This view requires a callable model to visualize'}]
      };
    public render() {
        this.el.style.cssText = "width: 100%";
        let root_element = document.createElement("div");
        root_element.style.cssText = "width: 100%;";
        this.model.on('change:response', this.resolvePromise, this);
        const data = this.model.get('value');
        ReactDOM.render(<ExplanationDashboard
            modelInformation={{modelClass: 'blackbox'} as any}
            dataSummary={{featureNames: data.featureNames, classNames: data.classNames}}
            testData={data.trainingData}
            predictedY={data.predictedY}
            probabilityY={data.probabilityY}
            trueY={data.trueY}
            precomputedExplanations={{
                localFeatureImportance: data.localExplanations,
                globalFeatureImportance: data.globalExplanation,
                ebmGlobalExplanation: data.ebmGlobalExplanation,
                customVis: data.customVisual}}
            stringParams={{contextualHelp: this.messages as any}}
            requestPredictions={this.invokePredictor.bind(this)}
        />, root_element);
        this.el.appendChild(root_element)
    }

    private invokePredictor(data: any[], abortSignal: AbortSignal): Promise<any[]> {
        const promise = new Promise<any[]>((resolve, reject) => {
            const requestIndex = this.requestIndex;
            this.requestIndex++;
            // handle timeout (set to 3 minutes)
            const timeout = window.setTimeout(() => {
                if (this.promiseDict[requestIndex]){
                    this.promiseDict[requestIndex].reject(new DOMException('Timeout: took longer than 3 minutes to process', 'TimeoutError'));
                    delete this.promiseDict[requestIndex];
                }
            }, 180000);
            this.promiseDict[requestIndex] = {resolve, reject, timeout};
            this.model.set('request', {id: requestIndex, data});
            this.touch();

            // handle abort
            abortSignal.addEventListener('abort', () => {
                clearTimeout(timeout);
                reject(new DOMException('Aborted', 'AbortError'));
                delete this.promiseDict[requestIndex];
            });
        })
        return promise;
    }

    private resolvePromise(): void {
        const response = this.model.get('response');
        if (response === undefined || response.id === undefined) {
            return;
        }
        const promise = this.promiseDict[response.id];
        if (promise === undefined) {
            return;
        }
        if (response.data === undefined || !Array.isArray(response.data)) {
            promise.reject('Null response');
        }
        else if (response.error !== undefined) {
            promise.reject(new DOMException(response.error, 'PythonError'));
        }
        else {
            promise.resolve(response.data);
        }
        clearTimeout(promise.timeout);
        delete this.promiseDict[response.id];
    }
};
