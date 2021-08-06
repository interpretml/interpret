<script>
</script>

<style type='text/scss'>
  @import 'define';

  #header {
    height: 50px;
    display: flex;
    padding: 0 20px;
    align-items: center;
    background: $blue-dark;
    justify-content: space-between;
  }

  #logo {
    display: flex;
    align-items: baseline;
  }

  #logo-text {
    font-size: 2rem;
    color: white;
    font-family: 'Exo';
    font-weight: 700;
    margin-right: 10px;
  }

  #logo-icon {
    width: 250px;
    display: flex;
  }

  #logo-tagline {
    font-size: 1.5rem;
    color: $brown-light;
    font-family: 'Neucha';
  }

  .icons {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    color: white;

    :not(:last-child) {
      margin-right: 15px;
    }
  }

  .icon {
    font-size: 1.7rem;
    display: flex;
    align-items: center;
    color: white;
  }

</style>

<div id="header">

  <div id="logo">
    <!-- <div id="logo-text">
      GAM Changer
    </div> -->
    
    <div id='logo-icon'>
      <img src='/img/logo.svg' alt='logo img'>
    </div>

    <!-- <div id='logo-tagline'>
      Exploring transformer models in your browser! 
    </div> -->
  </div>

  <div class="icons">

    <a target="_blank" href="https://github.com/interpretml/interpret">
      <div class="icon" title="Open-source code">
        <i class="fab fa-github"></i>
      </div>
    </a>

  </div>
</div>