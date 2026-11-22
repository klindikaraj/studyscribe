(function () {
  "use strict";
  var audio = document.getElementById("player");
  var noAudio = document.getElementById("no-audio");
  var audioError = document.getElementById("audio-error");
  var pending = null;

  function seek(seconds, play) {
    if (!audio) { return; }
    if (audio.readyState < 1) { pending = { seconds: seconds, play: play }; return; }
    audio.currentTime = seconds;
    if (play) {
      var started = audio.play();
      if (started && started.catch) { started.catch(function () {}); }
    }
  }

  function fromHash() {
    var found = /^#t=(\d+)$/.exec(window.location.hash);
    if (found) { seek(Number(found[1]), false); }
  }

  if (audio) {
    audio.addEventListener("loadedmetadata", function () {
      if (noAudio) { noAudio.hidden = true; }
      if (audioError) { audioError.hidden = true; }
      if (pending) { var wanted = pending; pending = null; seek(wanted.seconds, wanted.play); }
    });
    audio.addEventListener("error", function () {
      if (audioError && noAudio && noAudio.hidden) { audioError.hidden = false; }
    });
  }

  document.addEventListener("click", function (event) {
    var link = event.target.closest ? event.target.closest("a.ts") : null;
    if (!link) { return; }
    event.preventDefault();
    var seconds = Number(link.getAttribute("data-t"));
    if (!isFinite(seconds)) { return; }
    seek(seconds, true);
    try { window.history.replaceState(null, "", "#t=" + seconds); } catch (ignored) { /* file: */ }
  });

  window.addEventListener("hashchange", fromHash);
  fromHash();
})();
