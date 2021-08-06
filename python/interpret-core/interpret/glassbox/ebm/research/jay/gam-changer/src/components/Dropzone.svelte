<script>
  import * as d3 from 'd3';
  import { onMount } from 'svelte';
  import { splitFileName } from '../utils/utils';

  import oneIconSVG from '../img/one-icon.svg';
  import twoIconSVG from '../img/two-icon.svg';

  // 'sampleData' or 'modelData'
  export let dataType = 'sampleData';
  export let sidebarStore;

  let component = null;
  let inputElem = null;
  let isDragging = false;
  let dragElement = null;

  let errorMessage = ' ';

  // Bind the SVGs
  const preProcessSVG = (svgString) => {
    return svgString.replaceAll('black', 'currentcolor')
      .replaceAll('fill:none', 'fill:currentcolor')
      .replaceAll('stroke:none', 'fill:currentcolor');
  };

  /**
   * Dynamically bind SVG files as inline SVG strings in this component
   */
  const bindInlineSVG = () => {
    d3.select(component)
      .selectAll('.svg-icon.icon-one')
      .html(preProcessSVG(oneIconSVG));

    d3.select(component)
      .selectAll('.svg-icon.icon-two')
      .html(preProcessSVG(twoIconSVG));
  };

  const clickHandler = () => {
    inputElem.click();
  };

  const dragEnterHandler = (e) => {
    e.preventDefault();

    isDragging = true;

    // Store the drag element, so we don't leave the drag state when user hovers
    // over the messages
    dragElement = e.target;
  };

  const dragOverHandler = (e) => {
    e.preventDefault();
  };

  const dragLeaveHandler = (e) => {
    e.preventDefault();

    if (dragElement === e.target) {
      isDragging = false;
    }
  };

  const readJSON = (file) => {
    return new Promise((resolve, reject) => {
      let fr = new FileReader();  
      fr.onload = () => {
        resolve(JSON.parse(fr.result));
      };
      fr.onerror = reject;
      fr.readAsText(file);
    });
  };

  const validateFile = async (file) => {

    // Test if it is a valid .gamchanger file
    let extension = splitFileName(file)[1];
    let isGamchangerFile = extension === 'gamchanger';

    if (file.type !== 'application/json' && !isGamchangerFile) {
      errorMessage = 'It is not a JSON file';
      return false;
    }

    // Try to read the file
    let data = await readJSON(file);

    // Test if it is a valid file
    let isValid = false;
    if (dataType === 'sampleData') {
      isValid = (data.featureNames !== undefined && data.featureTypes !== undefined &&
        data.samples !== undefined && data.labels !== undefined);
    }

    if (dataType === 'modelData') {
      isValid = (data.intercept !== undefined && data.features !== undefined &&
        data.labelEncoder !== undefined && data.scoreRange !== undefined);
    }

    if (isGamchangerFile) {
      isValid = (data.modelData !== undefined && data.sampleData !== undefined);
    }

    if (!isValid) {
      if (isGamchangerFile) {
        errorMessage = 'Not a valid .gamchanger document file';
      } else {
        errorMessage = `Not a valid ${dataType === 'sampleData' ? 'sample data' : 'model data'} file`;
      }
      return false;
    }

    return data;
  };

  const dropHandler = async(e) => {
    e.preventDefault();
    let data = false;
    let file = null;

    if (e.dataTransfer.items) {
      // Use DataTransferItemList interface to access the file(s)
      if (e.dataTransfer.items[0].kind === 'file') {
        file = e.dataTransfer.items[0].getAsFile();
        data = await validateFile(file);
      }
    } else {
      // Use DataTransfer interface to access the file(s)
      file = e.dataTransfer.files[0];
      data = await validateFile(file);
    }

    if (!data) return;

    // Check if the file is .gamchanger file
    const extension = splitFileName(file)[1];
    const isGamchangerFile = extension === 'gamchanger';

    // Pass the data to other components through store
    sidebarStore.update(value => {
      value.curGroup = isGamchangerFile ? 'gamchangerCreated' : `${dataType}Created`;
      value.loadedData = data;
      return value;
    });

    isDragging = false;
  };

  const inputChanged = async (e) => {
    e.preventDefault();
    let data = false;

    let file = e.target.files[0];
    data = await validateFile(file);

    if (!data) return;

    // Pass the data to other components through store
    sidebarStore.update(value => {
      value.curGroup = `${dataType}Created`;
      value.loadedData = data;
      return value;
    });
  };

  const inputClicked = () => {
    console.log('input clicked');
  };

  onMount(() => {bindInlineSVG();});

</script>

<style type='text/scss'>

  @import '../define';

  .dropzone-tab {
    height: 100%;
    width: 100%;
    padding: 10px;
    display: flex;
    background-color: hsl(20, 16%, 99%);
  }

  .dropzone {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 3px dashed hsla(0, 0%, 86%);
    border-radius: 3px;
    transition: border 300ms ease-in-out;
    padding: 0 10px;
    gap: 18px;

    &.drag-over {
      border: 3px dashed $blue-300;
    }
  }

  .svg-icon {
    color: hsl(0, 0%, 90%);
    fill: hsl(0, 0%, 90%);
    display: inline-flex;
    align-items: center;
    pointer-events: none;

    :global(svg) {
      width: 40px;
      height: 40px;
    }
  }

  .drop-message {
    text-align: center;
    color: hsl(0, 0%, 50%);
    cursor: default;
  }

  .help-message {
    text-align: center;
    font-size: 0.9em;
    color: hsl(0, 0%, 50%);
    padding: 2px 8px;
    background: change-color($blue-dark, $alpha: 0.1);
    border-radius: 5px;

    &:hover {
      background: change-color($blue-dark, $alpha: 0.2);

      a {
        text-decoration: none;
      }
    }

    a {
      font-style: italic;
      color: hsl(0, 0%, 35%);
    }
  }

  .error-message {
    font-size: 0.9em;
    color: hsl(0, 50%, 56%);
    text-align: center;
    white-space: pre-wrap;
  }

</style>

<div class='dropzone-tab' bind:this={component}>

  <div class='dropzone'
    class:drag-over={isDragging}
    on:click={clickHandler}
    on:dragenter={dragEnterHandler}
    on:dragover={dragOverHandler}
    on:dragleave={dragLeaveHandler}
    on:drop={dropHandler}
  >
    {#if dataType === 'modelData'}
      <div class='svg-icon icon-one'></div>
    {:else}
      <div class='svg-icon icon-two'></div>
    {/if}
    
    <div class='drop-message'>
      {#if dataType === 'modelData'}
        Drop a <u>model file</u> (.json) or a <u>GAM Changer document</u> (.gamchanger) here to start
      {:else}
        Drop a <u>sample data file</u> (.json) file here to start
      {/if}
    </div>

    <div class='help-message' on:click={(e) => {e.stopPropagation();}}>
      <a href='https://gist.github.com/xiaohk/875b5374840c66689eb42a4b8820c3b5' target="_blank">How to generate this file?</a>
    </div>

    <span class='error-message'>
      {errorMessage}
    </span>

    <input
      accept="json"
      type="file"
      autocomplete="off"
      on:change={inputChanged}
      on:click={inputClicked}
      bind:this={inputElem}
      style="display: none;"/>

  </div>

</div>