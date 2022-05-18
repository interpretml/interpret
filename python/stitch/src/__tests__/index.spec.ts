// Copyright (c) Jupyter Development Team.
// Distributed under the terms of the Modified BSD License.

// Add any needed widget imports here (or from controls)
// import {} from '@jupyter-widgets/base';

import { createTestModel } from './utils';

import { StitchModel } from '..';

describe('Stitch', () => {
  describe('StitchModel', () => {
    it('should be createable', () => {
      const model = createTestModel(StitchModel);
      expect(model).toBeInstanceOf(StitchModel);
      expect(model.get('kernelmsg')).toEqual('');
      expect(model.get('clientmsg')).toEqual('');
      expect(model.get('srcdoc')).toEqual('<p>srcdoc should be defined by the user</p>');
      expect(model.get('initial_height')).toEqual('1px');
      expect(model.get('initial_width')).toEqual('1px');
      expect(model.get('initial_border')).toEqual('0');
    });

    it('should be createable with a value', () => {
      const state = { srcdoc: '<h1>New html!</h1>' };
      const model = createTestModel(StitchModel, state);
      expect(model).toBeInstanceOf(StitchModel);
      expect(model.get('kernelmsg')).toEqual('');
      expect(model.get('clientmsg')).toEqual('');
      expect(model.get('srcdoc')).toEqual('<h1>New html!</h1>');
      expect(model.get('initial_height')).toEqual('1px');
      expect(model.get('initial_width')).toEqual('1px');
      expect(model.get('initial_border')).toEqual('0');
    });
  });
});

// assert w.kernelmsg == ''
// assert w.clientmsg == ''
// assert w.srcdoc == '<p>srcdoc should be defined by the user</p>'
// assert w.initial_height == '1px'
// assert w.initial_width == '1px'
// assert w.initial_border == '0'
// 