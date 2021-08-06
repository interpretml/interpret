<script>
  import * as d3 from 'd3';
  import { onMount, createEventDispatcher } from 'svelte';
  import selectIconSVG from '../img/select-icon.svg';
  import dragIconSVG from '../img/drag-icon.svg';

  import { tooltipConfigStore } from '../store';

  export let name = '';

  let selectMode = false;
  let component = null;
  let tooltipConfig = {};
  let mouseoverTimeout = null;
  const dispatch = createEventDispatcher();

  tooltipConfigStore.subscribe(value => {tooltipConfig = value;});

  /**
   * Event handler for the select button in the header
   */
  const selectModeSwitched = () => {
    selectMode = !selectMode;
    dispatch('selectModeSwitched', {
      selectMode: selectMode
    });
  };

  export const reset = () => {
    d3.select(component)
      .select(`#${name}-toggle`)
      .property('checked', false);
    selectMode = false;
  };

  /**
   * Dynamically bind SVG files as inline SVG strings in this component
   */
  const bindInlineSVG = () => {
    d3.select(component)
      .select('.svg-icon#toggle-button-move')
      .html(dragIconSVG.replaceAll('black', 'currentcolor'));

    d3.select(component)
      .select('.svg-icon#toggle-button-select')
      .html(selectIconSVG.replaceAll('black', 'currentcolor'));
  };


  const mouseoverHandler = (e, message, width, yOffset) => {
    let node = e.currentTarget;
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
    }, 1000);
  };

  const mouseleaveHandler = () => {
    clearTimeout(mouseoverTimeout);
    mouseoverTimeout = null;
    tooltipConfig.show = false;
    tooltipConfigStore.set(tooltipConfig);
  };

  onMount(() => {
    bindInlineSVG();
  });

</script>

<style type='text/scss'>
  @import '../define';

  $secondary-color: hsl(0, 0%, 40%);
  $border-radius: 13px;
  $dot-background: $blue-dark;

  @mixin toggle-button-label {
    padding: 5px 0;
    width: 50%;
    display: flex;
    justify-content: center;
    position: absolute;
    z-index: 2;
    color: white;
    transition: all .2s ease;
  }

  .toggle-wrapper {
    display: flex;
    // Expect the parent to have a width defined
    width: 100%;
  }

  .toggle {
    display: none;
    
    &:checked + .toggle-button .dot {
      left: 50%;
    }
  }

  .toggle-button {
    outline: 0;
    display: flex;
    align-items: center;
    justify-content: space-around;
    width: 100%;
    position: relative;
    height: 2em;
    cursor: pointer;
    user-select: none;

    background: $brown-50;
    border: 1px solid hsl(0, 0%, 86%);
    border-radius: $border-radius;
    transition: all .4s ease;

    .left-label {
      @include toggle-button-label;
      left: 0;

      &.select-mode {
        color: $secondary-color;
      }
    }

    .right-label {
      @include toggle-button-label;
      right: 0px;
      color: $secondary-color;

      &.select-mode {
        color: white;
      }
    }

    .dot {
      position: absolute;
      width: 50%;
      height: 100%;
      left: 0;
      border-radius: $border-radius;
      background: $dot-background;
      transition: all .2s ease;
    }

    .icon-label {
      margin-left: 5px;
    }
  }

  :global(.toggle-wrapper .svg-icon) {
    display: flex;
    justify-content: center;
    align-items: center;

    :global(svg) {
      width: 1.2em;
      height: 1.2em;
    }
  }

</style>

<div class='toggle-wrapper' bind:this={component}>

  <input class='toggle' id={`${name}-toggle`} type='checkbox' on:change={selectModeSwitched}/>

  <label for={`${name}-toggle`} class='toggle-button'>

    <div class='left-label' class:select-mode = {selectMode}
      on:mouseenter={(e) => mouseoverHandler(e, 'navigate graph', 120, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon' id='toggle-button-move'></div>
      <div class='icon-label'>Move</div>
    </div>

    <div class='right-label' class:select-mode = {selectMode}
      on:mouseenter={(e) => mouseoverHandler(e, 'select nodes', 110, 30)}
      on:mouseleave={mouseleaveHandler}
    >
      <div class='svg-icon' id='toggle-button-select'></div>
      <div class='icon-label'>Select</div>
    </div>

    <div class='dot'></div>

  </label>
</div>
