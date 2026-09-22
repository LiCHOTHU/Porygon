# Animation validation

- Passed: Restoration matches the negative gradient of the stated potential inside and outside the soft radius.
- Passed: A03 critic cap is applied before guidance scaling; the final target displacement obeys its cap.
- Passed: Both actors share initialization and noise; every fixed-target fitting step reduces MSE; every target step respects the total cap.
- Passed: The recorded toy trajectories reproduce the illustrated drift-versus-restoration comparison.
- Passed: Measured ablation source is unchanged; displayed means recompute to 67.28, 0, 0, 65.74, and 70.34 percent.
- Passed: Five MP4s fully decode, match their 30 fps / 720p / H.264 durations, and match the files embedded in PowerPoint.
- Passed: All five PowerPoint video timing nodes specify automatic start at slide entry.
- Passed: The silent method preview fully decodes and lasts 120 seconds at 1920 × 1080.

The fixed-critic toy is illustrative. PowerPoint XML and media are checked; native PowerPoint slideshow playback is not available in this environment.

## Browser playback check

Verified in local headless Google Chrome: all five slide clips loaded and sought to the requested time; a user-gesture click started playback; playback time advanced; the full-screen stage retained its 16:9 geometry; all five standalone gallery entries were available. No video decode error occurred.
