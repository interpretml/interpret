<script>
  import * as d3 from 'd3';
  import { onMount, createEventDispatcher } from 'svelte';
  import { round } from '../utils/utils';

  // Stores
  import { tooltipConfigStore } from '../store';

  // SVGs
  import { bindInlineSVG } from '../utils/svg-icon-binding';

  export let controlInfo = undefined;
  /**
   * Different versions of the context menu bar
   * 'cont': show move, increasing, decreasing, interpolation, merge, delete
   * 'cat': show move, merge, delete
   */
  export let type = 'cont';

  $: controlInfo, (() => {
    if (controlInfo.toSwitchMoveMode) {
      switchMoveMode();
      controlInfo.toSwitchMoveMode = false;
    }
  })();

  // Component variables
  let component = null;
  let tooltipConfig = {};
  let mouseoverTimeout = null;

  // Initialized when mounted
  let width = null;
  let height = null;

  tooltipConfigStore.subscribe(value => {tooltipConfig = value;});

  const dispatch = createEventDispatcher();

  const prepareSubMenu = (newMode) => {
    if (controlInfo.subItemMode !== null && controlInfo.subItemMode !== newMode) {
    // Hide the confirmation panel
      hideConfirmation(controlInfo.subItemMode);

      // Exit the sub-item mode
      controlInfo.subItemMode = null;
    }

    controlInfo.subItemMode = newMode;

    hideToolTipDuringSubMenu();
  };

  const inputChanged = (e) => {
    prepareSubMenu('change');

    e.preventDefault();
    let value = parseFloat(e.target.value);

    if (isNaN(value)) value = 0;
    controlInfo.setValue = value;

    dispatch('inputChanged');
  };

  const inputAdd = () => {
    prepareSubMenu('change');

    if (controlInfo.setValue === null) {
      controlInfo.setValue = 0;
    } else {
      controlInfo.setValue += controlInfo.changeUnit;
      controlInfo.setValue = round(controlInfo.setValue, 3);
    }

    dispatch('inputChanged');
  };

  const inputMinus = () => {
    prepareSubMenu('change');

    if (controlInfo.setValue === null) {
      controlInfo.setValue = 0;
    } else {
      controlInfo.setValue -= controlInfo.changeUnit;
      controlInfo.setValue = round(controlInfo.setValue, 3);
    }

    dispatch('inputChanged');
  };

  const switchMoveMode = () => {
    if (controlInfo.moveMode) {
      // Shrink the menu bar to make space for action
      d3.select(component)
        .transition()
        .duration(300)
        .ease(d3.easeCubicInOut)
        .style('width', '120px')
        .on('end', () => {
          d3.select(component)
            .selectAll('div.collapse-item')
            .style('display', 'flex');
        });

    } else {
      // restore the menu bar width
      d3.select(component)
        .selectAll('div.collapse-item')
        .style('display', 'none');

      d3.select(component)
        .transition()
        .duration(300)
        .ease(d3.easeCubicInOut)
        .style('width', `${width}px`);
    }
  };

  const moveButtonClicked = () => {
    prepareSubMenu(null);

    controlInfo.moveMode = !controlInfo.moveMode;

    switchMoveMode();

    dispatch('moveButtonClicked');
  };

  const moveCheckClicked = () => {

    controlInfo.moveMode = !controlInfo.moveMode;
    switchMoveMode();
    hideToolTipDuringSubMenu();

    dispatch('moveCheckClicked');
  };

  const moveCancelClicked = () => {
    controlInfo.moveMode = !controlInfo.moveMode;
    switchMoveMode();
    hideToolTipDuringSubMenu();

    dispatch('moveCancelClicked');
  };

  const increasingClicked = () => {
    prepareSubMenu('increasing');

    dispatch('increasingClicked');
  };

  const decreasingClicked = () => {
    // Need to handle the case where people change mode without checking/crossing
    // We don't need to recover the original graph then enter new mode preview
    // We can directly enter the new mode with animation
    prepareSubMenu('decreasing');

    dispatch('decreasingClicked');
  };

  const interpolationClicked = () => {
    prepareSubMenu('interpolation');

    dispatch('interpolationClicked');
  };

  const interpolateMinusClicked = (e) => {
    e.stopPropagation();
    if (controlInfo.step - 1 !== 0) {
      controlInfo.step --;
      controlInfo.interpolationMode = 'equal';
      dispatch('interpolateUpdated');
    }
  };

  const interpolatePlusClicked = (e) => {
    e.stopPropagation();
    if (controlInfo.step + 1 !== 21) {
      controlInfo.step ++;
      controlInfo.interpolationMode = 'equal';
      dispatch('interpolateUpdated');
    }
  };

  const interpolateInplaceClicked = (e) => {
    e.stopPropagation();
    controlInfo.interpolationMode = 'inplace';
    dispatch('interpolateUpdated');
  };

  const interpolateEqualClicked = (e) => {
    e.stopPropagation();
    controlInfo.interpolationMode = 'equal';
    dispatch('interpolateUpdated');
  };

  const interpolateRegressionClicked = (e) => {
    e.stopPropagation();
    controlInfo.interpolationMode = 'regression';
    dispatch('interpolateUpdated');
  };

  const interpolateTextClicked = (e) => {
    e.stopPropagation();
    controlInfo.interpolationMode = 'equal';
    dispatch('interpolateUpdated');
  };

  const mergeClicked = () => {
    prepareSubMenu('merge');
    controlInfo.mergeMode = 'left';
    dispatch('mergeClicked');
  };

  const mergeToLeftClicked = (e) => {
    e.stopPropagation();
    controlInfo.mergeMode = 'left';
    dispatch('mergeUpdated');
  };

  const mergeToAverageClicked = (e) => {
    e.stopPropagation();
    controlInfo.mergeMode = 'average';
    dispatch('mergeUpdated');
  };

  const mergeToRightClicked = (e) => {
    e.stopPropagation();
    controlInfo.mergeMode = 'right';
    dispatch('mergeUpdated');
  };

  const deleteClicked = () => {
    prepareSubMenu('delete');

    dispatch('deleteClicked');
  };

  const hideToolTipDuringSubMenu = () => {
    // hide the tooltip
    clearTimeout(mouseoverTimeout);
    mouseoverTimeout = null;
    tooltipConfig.show = false;
    tooltipConfig.hideAnimated = false;
    tooltipConfigStore.set(tooltipConfig);
  };

  const subItemCheckClicked = (e) => {
    e.stopPropagation();
    hideToolTipDuringSubMenu();
    dispatch('subItemCheckClicked');
  };

  const subItemCancelClicked = (e) => {
    e.stopPropagation();
    hideToolTipDuringSubMenu();
    dispatch('subItemCancelClicked');
  };

  export const showConfirmation = (option, delay=500) => {
    d3.timeout(() => {
      let componentSelect = d3.select(component);

      componentSelect.select('.items')
        .style('overflow', 'visible');
      
      componentSelect.select(`.sub-item-${option}`)
        .classed('hidden', false);
    }, delay);
  };

  export const hideConfirmation = (option, delay=0) => {
    const _hideConfirmation = (option) => {
      let componentSelect = d3.select(component);

      componentSelect.select('.items')
        .style('overflow', 'hidden');
      
      componentSelect.select(`.sub-item-${option}`)
        .classed('hidden', true);
    };

    if (delay > 0) {
      d3.timeout(() => _hideConfirmation(option), delay);
    } else {
      _hideConfirmation(option);
    }
  };

  const mouseoverHandler = (e, message, width, yOffset) => {
    e.stopPropagation();
    let node = e.currentTarget;

    // Do not show tooltip in sub-menu mode if hovering over the sub-menu items
    if (controlInfo.subItemMode !== null) {
      const target = d3.select(e.explicitOriginalTarget);
      if (target.size() > 0) {
        if (!target.classed('show-tooltip')) {
          if (target.classed('sub-item-child') || target.classed('sub-item')) {
            return;
          }
        }
      }
    }

    mouseoverTimeout = setTimeout(() => {
      let position = node.getBoundingClientRect();
      let curWidth = position.width;

      let tooltipCenterX = position.x + curWidth / 2;
      let tooltipCenterY = position.y - yOffset;
      tooltipConfig.html = `
        <div class='tooltip-content' style='display: flex; flex-direction: column;
          justify-content: center;'>
          ${message}
        </div>
      `;
      tooltipConfig.width = width;
      tooltipConfig.maxWidth = width;
      tooltipConfig.left = tooltipCenterX - tooltipConfig.width / 2;
      tooltipConfig.top = tooltipCenterY;
      tooltipConfig.fontSize = '14px';
      tooltipConfig.show = true;
      tooltipConfig.orientation = 's';
      tooltipConfigStore.set(tooltipConfig);
    }, 400);
  };

  export const mouseleaveHandler = () => {
    clearTimeout(mouseoverTimeout);
    mouseoverTimeout = null;
    tooltipConfig.show = false;
    tooltipConfigStore.set(tooltipConfig);
  };

  onMount(() => {
    bindInlineSVG(component);
    // Get the width of this bar
    let bbox = component.getBoundingClientRect();
    width = bbox.width;
    height = bbox.height;

    // Register the width as html data, so we can access it later (the computed
    // width can shrink during move mode)
    component.setAttribute('data-max-width', width);
  });

</script>

<style type='text/scss'>
  @import '../define';

  $secondary-color: hsl(0, 0%, 40%);
  $border-radius: 13px;
  $dot-background: $blue-dark;
  $hover-color: $blue-dark;
  $check-hover-color: $blue-600;

  @mixin item-input-arrow {
    position: absolute;
    display: flex;
    align-content: center;
    justify-content: center;
    color: $indigo-dark;
    width: 12px;
    height: 25px;
    right: 0;

    &:hover {
      color: $hover-color;
    }
  }

  .menu-wrapper {
    height: 50px;
    border-radius: 4px;
    width: 100%;
    background: white;
    box-shadow: 0px 4px 16px hsla(245, 100%, 11%, 0.12);
  }

  .items {
    display: flex;
    align-items: center;
    padding: 0 5px;
    overflow: hidden;
    width: 100%;
    height: 100%;
  }

  .item {
    width: 40px;
    height: 50px;
    display: flex;
    cursor: pointer;
    justify-content: center;
    align-items: center;
    position: relative;
    flex-shrink: 0;

    color: $indigo-dark;

    &.has-input {
      width: 80px;
      justify-content: flex-start;
    }

    &.selected {
      color: $blue-reg;

      &:hover {
        color: $blue-reg;
      }
    }

    &.disabled {
      cursor: no-drop;
      pointer-events: none;
    }

    &:hover {
      color: $hover-color;
    }
  }

  .item-input {
    text-align: center;
    font-size: 1em;
    line-height: 35px;
    height: 35px;
    max-width: 65px;
    color: $indigo-dark;
    opacity: 0.8;
    border-radius: 2px;
    border: 1px solid $gray-300;
    outline: none;

    &:focus {
      border-radius: 2px;
      border: 1px solid hsla(0, 0%, 0%, 0.2);
    }
  }

  .collapse-item {
    display: none;
    flex-shrink: 0;

    .item {
      width: 30px;

      .svg-icon {
        color: $indigo-dark;

        :global(svg) {
          width: 1em;
          height: 1em;
        }
      }

      .svg-icon.icon-check {
        color: $indigo-dark;
      }

      &:hover {
        color: $blue-600;

        .svg-icon {
          color: currentcolor;
        }

        .svg-icon.icon-check {
          color: $check-hover-color;
        }
      }
    }
  }

  .sub-item {
    position: absolute;
    left: 50%;
    top: 50px;
    height: 30px;
    padding: 0 5px;
    z-index: 1;
    transform: translate(-50%);
    background: white;

    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 2px 6px 2px hsla(205, 5%, 25%, 0.15);
    border-radius: 4px;

    .item {
      width: 30px;
      height: 30px;

      .svg-icon {
        color: $indigo-dark;

        :global(svg) {
          width: 1em;
          height: 1em;
        }
      }

      &:hover {
        color: $blue-600;

        .svg-icon {
          color: currentcolor;
        }

        .icon-check.svg-icon {
          color: $check-hover-color;
        }
      }
    }

    &.hidden {
      visibility: hidden;
      pointer-events: none;
    }
  }

  .hidden {
    .item {
      visibility: hidden;
      pointer-events: none;
    }
  }

  .sub-item-child.selected {
    .svg-icon {
      color: $blue-reg;
    }
  }

  .svg-icon.item-input-up {
    @include item-input-arrow;

    align-items: flex-end;
    padding-bottom: 5px;

    top: 0;
  }

  .interpolate-step {
    display: flex;
    align-items: center;
    border: 2px solid $gray-200;
    border-radius: 4px;
    height: 24px;
    margin: 0 5px;

    &.selected {
      border: 2px solid change-color($blue-reg, $lightness: 70%);
      
      .sub-item-child .svg-icon {
        color: $blue-reg;

        &:hover {
          color: $blue-reg;
        }
      }
    }
  }

  .sub-item-child {

    .icon-check.svg-icon {
      color: $indigo-dark;
    }

  }

  .item-step-text {
    border-right: 2px solid $gray-200;
    border-left: 2px solid $gray-200;
    height: 24px;
    width: 30px;
    color: $indigo-dark;
    text-align: center;

    &:hover {
      color: $blue-600;
    }

    &.selected {
      color: $blue-reg;
      border-right: 2px solid change-color($blue-reg, $lightness: 70%);
      border-left: 2px solid change-color($blue-reg, $lightness: 70%);
    }
  }

  .svg-icon.item-input-down {
    @include item-input-arrow;

    align-items: flex-start;
    padding-top: 5px;

    bottom: 0;
  }

  .separator {
    margin: 0 5px;
    width: 1px;
    background-color: hsl(0, 0%, 90%);
    height: 100%;
    flex-shrink: 0;
  }

  :global(.menu-wrapper .svg-icon) {
    display: flex;
    justify-content: center;
    align-items: center;

    :global(svg) {
      width: 1.2em;
      height: 1.2em;
      fill: currentcolor;
      stroke: currentcolor;
    }
  }

  :global(.menu-wrapper .has-input .svg-icon svg) {
    width: 10px;
    height: 5px;
  }

  .hidden {
    visibility: hidden;
    pointer-events: none;
  }

</style>

<div class='menu-wrapper' bind:this={component}>

  <div class='items'>

    <!-- Move button -->
    <div class='item' on:click={moveButtonClicked}
      class:selected={controlInfo.moveMode}
      class:disabled={controlInfo.moveMode}
      on:mouseenter={(e) => mouseoverHandler(e, 'move', 55, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-updown'></div>
    </div>

    <div class='separator'></div>

    <div class='collapse-item'>
      <!-- Check button -->
      <div class='item' on:click={moveCheckClicked}
        on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
        on:mouseleave={mouseleaveHandler}
      >
        <div class='svg-icon icon-check'></div>
      </div>

      <!-- Cancel button -->
      <div class='item' on:click={moveCancelClicked}
        on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
        on:mouseleave={mouseleaveHandler}
      >
        <div class='svg-icon icon-refresh'></div>
      </div>

    </div>

    {#if type === 'cont'}

    <!-- Increasing -->
    <div class='item'
      class:selected={controlInfo.subItemMode==='increasing'}
      on:click={increasingClicked}
      on:mouseenter={(e) => mouseoverHandler(e, 'monotonically increasing', 120, 52)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-increasing'></div>
      
      <div class='sub-item sub-item-increasing hidden'>
        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>

      </div>
    </div>

    <!-- Decreasing -->
    <div class='item'
      class:selected={controlInfo.subItemMode==='decreasing'}
      on:click={decreasingClicked}
      on:mouseenter={(e) => mouseoverHandler(e, 'monotonically decreasing', 120, 52)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-decreasing'></div>

      <div class='sub-item sub-item-decreasing hidden'>
        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>

      </div>
    </div>

    <div class='separator'></div>

    <!-- Interpolation -->
    <div class='item'
      class:selected={controlInfo.subItemMode==='interpolation'}
      on:click={interpolationClicked}
      on:mouseenter={(e) => mouseoverHandler(e, 'interpolate', 85, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-interpolate'></div>

      <div class='sub-item sub-item-interpolation hidden'>

        <!-- Inplace interpolation button -->
        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='interpolation' && controlInfo.interpolationMode === 'inplace'}
          on:mouseenter={(e) => mouseoverHandler(e, 'inplace interpolation', 95, 52)}
          on:mouseleave={mouseleaveHandler}
          on:click={interpolateInplaceClicked}
        >
          <div class='svg-icon icon-inplace'></div>
        </div>

        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='interpolation' && controlInfo.interpolationMode === 'regression'}
          on:mouseenter={(e) => mouseoverHandler(e, 'inplace regression', 95, 52)}
          on:mouseleave={mouseleaveHandler}
          on:click={interpolateRegressionClicked}
        >
          <div class='svg-icon icon-regression'></div>
        </div>

        <div class='separator'></div>

        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='interpolation' && controlInfo.interpolationMode === 'equal'}
          on:mouseenter={(e) => mouseoverHandler(e, 'equalize bin size', 95, 52)}
          on:mouseleave={mouseleaveHandler}
          on:click={interpolateEqualClicked}
        >
          <div class='svg-icon icon-interpolation'></div>
        </div>

        <div class='interpolate-step'
          class:selected={controlInfo.subItemMode==='interpolation' && controlInfo.interpolationMode === 'equal'}
          on:mouseenter={(e) => mouseoverHandler(e, 'number of bins', 90, 52)}
          on:mouseleave={mouseleaveHandler}
        >
          <!-- Minus button -->
          <div class='item sub-item-child show-tooltip' on:click={interpolateMinusClicked}>
            <div class='svg-icon icon-minus'></div>
          </div>

          <!-- Interpolation step input -->
          <div class='item-step-text'
            on:click={interpolateTextClicked}
            class:selected={controlInfo.subItemMode==='interpolation' && controlInfo.interpolationMode === 'equal'}
            >
          {controlInfo.step}</div>

          <!-- Plus button -->
          <div class='item sub-item-child' on:click={interpolatePlusClicked}>
            <div class='svg-icon icon-plus'></div>
          </div>
        </div>

        <div class='separator'></div>

        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>
      </div>

    </div>

    <div class='separator'></div>

    {/if}

    <!-- Merge -->
    <div class='item'
      class:selected={controlInfo.subItemMode==='merge'}
      on:click={mergeClicked}
      on:mouseenter={(e) => mouseoverHandler(e, 'align', 60, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-merge'></div>

      <div class='sub-item sub-item-merge hidden'>

        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='merge' && controlInfo.mergeMode === 'left'}
          on:mouseenter={(e) => mouseoverHandler(e, 'to left', 70, 30)}
          on:mouseleave={mouseleaveHandler}
          on:click={mergeToLeftClicked}
        >
          <div class='svg-icon icon-merge'></div>
        </div>

        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='merge' && controlInfo.mergeMode === 'average'}
          on:mouseenter={(e) => mouseoverHandler(e, 'to average', 90, 30)}
          on:mouseleave={mouseleaveHandler}
          on:click={mergeToAverageClicked}
        >
          <div class='svg-icon icon-merge-average'></div>
        </div>

        <div class='item sub-item-child show-tooltip'
          class:selected={controlInfo.subItemMode==='merge' && controlInfo.mergeMode === 'right'}
          on:mouseenter={(e) => mouseoverHandler(e, 'to right', 70, 30)}
          on:mouseleave={mouseleaveHandler}
          on:click={mergeToRightClicked}
        >
          <div class='svg-icon icon-merge-right'></div>
        </div>

        <div class='separator'></div>

        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>

      </div>
    </div>

    <!-- <div class='separator'></div> -->

    <!-- Input field -->
    <div class='item has-input'
      on:mouseenter={(e) => mouseoverHandler(e, 'set scores', 90, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <input class='item-input'
        placeholder={'score'} bind:value={controlInfo.setValue}
        on:change={inputChanged}
      >

      <div class='svg-icon item-input-up icon-input-up'
        on:click={inputAdd}
      ></div>

      <div class='svg-icon item-input-down icon-input-down'
        on:click={inputMinus}
      ></div>

      <div class='sub-item sub-item-change hidden'>
        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>

      </div>

    </div>

    <div class='separator'></div>

    <!-- Deletion -->
    <div class='item'
      class:selected={controlInfo.subItemMode==='delete'}
      on:click={deleteClicked}
      on:mouseenter={(e) => mouseoverHandler(e, 'delete', 60, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon icon-delete'></div>

      <div class='sub-item sub-item-delete hidden'>
        <!-- Check button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCheckClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'commit', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-check'></div>
        </div>

        <!-- Cancel button -->
        <div class='item sub-item-child show-tooltip' on:click={subItemCancelClicked}
          on:mouseenter={(e) => mouseoverHandler(e, 'cancel', 65, 30)}
          on:mouseleave={mouseleaveHandler}
        >
          <div class='svg-icon icon-refresh'></div>
        </div>

      </div>
    </div>

  </div>

</div>
