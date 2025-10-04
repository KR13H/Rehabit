


socket.on("audio_cue", ({ audio }) => {
  const player = new Audio("data:audio/wav;base64," + audio);
  player.play().catch(e => console.error("Audio playback failed:", e));
});