import * as d3 from 'd3';

import mergeIconSVG from '../img/merge-icon.svg';
import mergeRightIconSVG from '../img/merge-right-icon.svg';
import mergeAverageIconSVG from '../img/merge-average-icon.svg';
import increasingIconSVG from '../img/increasing-icon.svg';
import decreasingIconSVG from '../img/decreasing-icon.svg';
import upDownIconSVG from '../img/updown-icon.svg';
import trashIconSVG from '../img/trash-icon.svg';
import trashCommitIconSVG from '../img/trash-commit-icon.svg';
import rightArrowIconSVG from '../img/right-arrow-icon.svg';
import locationIconSVG from '../img/location-icon.svg';
import upIconSVG from '../img/up-icon.svg';
import downIconSVG from '../img/down-icon.svg';
import interpolateIconSVG from '../img/interpolate-icon.svg';
import inplaceIconSVG from '../img/inplace-icon.svg';
import interpolationIconSVG from '../img/interpolation-icon.svg';
import regressionIconSVG from '../img/regression-icon.svg';
import thumbupIconSVG from '../img/thumbup-icon.svg';
import thumbupEmptyIconSVG from '../img/thumbup-empty-icon.svg';
import penIconSVG from '../img/pen-icon.svg';
import checkIconSVG from '../img/check-icon.svg';
import refreshIconSVG from '../img/refresh-icon.svg';
import minusIconSVG from '../img/minus-icon.svg';
import plusIconSVG from '../img/plus-icon.svg';
import originalSVG from '../img/original-icon.svg';

const preProcessSVG = (svgString) => {
  return svgString.replaceAll('black', 'currentcolor')
    .replaceAll('fill:none', 'fill:currentcolor')
    .replaceAll('stroke:none', 'fill:currentcolor');
};

/**
 * Dynamically bind SVG files as inline SVG strings in this component
 */
export const bindInlineSVG = (component) => {
  d3.select(component)
    .selectAll('.svg-icon.icon-merge')
    .html(preProcessSVG(mergeIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-merge-average')
    .html(mergeAverageIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-merge-right')
    .html(mergeRightIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-increasing')
    .html(increasingIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-decreasing')
    .html(decreasingIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-updown')
    .html(preProcessSVG(upDownIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-input-up')
    .html(preProcessSVG(upIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-input-down')
    .html(preProcessSVG(downIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-delete')
    .html(trashIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-commit-delete')
    .html(preProcessSVG(trashCommitIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-right-arrow')
    .html(preProcessSVG(rightArrowIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-location')
    .html(preProcessSVG(locationIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-interpolate')
    .html(interpolateIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-inplace')
    .html(inplaceIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-interpolation')
    .html(interpolationIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-regression')
    .html(regressionIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-thumbup')
    .html(preProcessSVG(thumbupIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-box')
    .html(preProcessSVG(thumbupEmptyIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-pen')
    .html(preProcessSVG(penIconSVG));

  d3.select(component)
    .selectAll('.svg-icon.icon-check')
    .html(checkIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-refresh')
    .html(refreshIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-minus')
    .html(minusIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-plus')
    .html(plusIconSVG.replaceAll('black', 'currentcolor'));

  d3.select(component)
    .selectAll('.svg-icon.icon-original')
    .html(originalSVG.replaceAll('black', 'currentcolor'));

};