import React from 'react';
import { ExplanationDashboard, FairnessDashboard } from '../../dashboard/rel';
import  {breastCancerData} from '../__mock_data/dummyData';
import {ibmData} from '../__mock_data/ibmData';
import {irisData} from '../__mock_data/irisData';
import {bostonData} from '../__mock_data/bostonData';

  var ibmNoClass = _.cloneDeep(ibmData);
  ibmNoClass.classNames = undefined;

  var irisNoFeatures = _.cloneDeep(irisData);
  irisNoFeatures.featureNames = undefined;

    class App extends React.Component {
      constructor(props) {
        super(props);
        this.state = {value: 3};
        this.handleChange = this.handleChange.bind(this);
        this.generateRandomScore = this.generateRandomScore.bind(this);
      }

      static choices = [
        {label: 'bostonData', data: bostonData},
        {label: 'irisData', data: irisData},
        {label: 'ibmData', data: ibmData},
        {label: 'breastCancer', data: breastCancerData},
        {label: 'ibmNoClass', data: ibmNoClass},
        {label: 'irisNoFeature', data: irisNoFeatures}
      ]

      messages = {
        'LocalExpAndTestReq': [{displayText: 'LocalExpAndTestReq'}],
        'LocalOrGlobalAndTestReq': [{displayText: 'LocalOrGlobalAndTestReq'}],
        'TestReq': [{displayText: 'TestReq'}],
        'PredictorReq': [{displayText: 'PredictorReq'}]
      }

      handleChange(event){
        this.setState({value: event.target.value});
      }

      generateRandomScore(data) {
        return Promise.resolve(data.map(x => Math.random()));
      }

      generateRandomProbs(classDimensions, data, signal) {
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve(data.map(x => Array.from({length:classDimensions}, (unused) => Math.random())))}, 300);
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });

        return promise;
      }

      generateExplanatins(explanations, data, signal) {
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve(explanations)}, 300);
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });

        return promise;
      }


      render() {
        const data = _.cloneDeep(App.choices[this.state.value].data);
        // data.localExplanations = undefined;
        const classDimension = data.localExplanations && Array.isArray(data.localExplanations[0][0]) ?
          data.localExplanations.length : 1;
        const newPredY = [1,2,3,4,5,6].map(unused => {
          return [...data.predictedY].map(val => {
            if (Math.random() > 0.8) {
              return 0;
            }
            return val;
          })
        })
        return (
          <div style={{backgroundColor: 'grey', height:'100%'}}>
            <label>
              Select dataset:
            </label>
            <select value={this.state.value} onChange={this.handleChange}>
              {App.choices.map((item, index) => <option key={item.label} value={index}>{item.label}</option>)}
            </select>
              <div style={{ width: '80vw', backgroundColor: 'white', margin:'50px auto'}}>
                  <div style={{ width: '100%'}}>
                      <FairnessDashboard
                        modelInformation={{modelClass: 'blackbox'}}
                        dataSummary={{featureNames: data.featureNames, classNames: data.classNames}}
                        testData={data.trainingData}
                        predictedY={newPredY}
                        probabilityY={data.probabilityY}
                        trueY={data.trueY}
                        precomputedExplanations={{localFeatureImportance: data.localExplanations}}
                        requestPredictions={this.generateRandomProbs.bind(this, classDimension)}
                        stringParams={{contextualHelp: this.messages}}
                        requestLocalFeatureExplanations={this.generateExplanatins.bind(this, App.choices[this.state.value].data.localExplanations)}
                        augmentedCount={2}
                        key={new Date()}
                      />
                  </div>
              </div>
          </div>
        );
      }
    }

    export default App;