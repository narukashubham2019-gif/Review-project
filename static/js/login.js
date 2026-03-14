(function(){
  // Page load animation
  window.addEventListener('load', function(){
    setTimeout(function(){
      document.getElementById('loader').classList.add('out');
      document.getElementById('card').classList.add('vis');
    }, 350);
  });

  var SK = 'rp_l';
  function lm(){ try{ return JSON.parse(localStorage.getItem(SK))||{}; }catch(e){ return {}; } }
  function sm(m){ try{ localStorage.setItem(SK, JSON.stringify(m)); }catch(e){} }

  // Restore remember-me username
  var meta = lm();
  if(meta.ru){ document.getElementById('username').value = meta.ru; document.getElementById('rem').checked = true; }

  // Password toggle
  document.getElementById('pBtn').addEventListener('click', function(){
    var p = document.getElementById('password');
    p.type = p.type==='password' ? 'text' : 'password';
    this.textContent = p.type==='password' ? '👁' : '🙈';
  });

  // Submit
  document.getElementById('lf').addEventListener('submit', function(e){
    var u = document.getElementById('username').value.trim();
    var p = document.getElementById('password').value;
    var ok = true;

    if(!u){ document.getElementById('uerr').style.display='block'; ok=false; } else { document.getElementById('uerr').style.display='none'; }
    if(!p){ document.getElementById('perr').style.display='block'; ok=false; } else { document.getElementById('perr').style.display='none'; }
    if(!ok){ e.preventDefault(); return; }

    // Save remember-me username
    var m = lm();
    if(document.getElementById('rem').checked) m.ru = u; else delete m.ru;
    sm(m);

    // Show loading spinner
    document.getElementById('sbtn').classList.add('ld');
    document.getElementById('sbtn').disabled = true;
  });
})();