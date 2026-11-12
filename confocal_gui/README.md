# Confocal alignment & scan GUI

Replaces the alignment and scan part of a Jupyter-notebook workflow with one window that
walks you through the workflow.

```
python run_confocal_gui.py            # real hardware (pyscan: BSC203 stage, BSC303 piezo, PicoHarp 300)
python run_confocal_gui.py --sim      # simulated microscope, for practice / testing
python run_confocal_gui.py --sim --sim-speed 20   # simulator, 20x faster
```

Needs: numpy, scipy, matplotlib, and one Qt binding (PyQt5, PySide6 or PyQt6, all come with Anaconda).
pyscan is only imported when real hardware is used (path set by `pyscan_path`).

## Tabs

Every panel divider can be dragged to resize. Short messages appear in the status bar at the bottom.

* **Move**: stage (coarse, min 10 um) and piezo (fine, min 0.1 um) jog buttons with step sizes, plus
  the current stage (mm) and piezo (um) positions. Arrows are labelled right = +x, up = +y; which hardware
  axis they move is set by `screen_right` / `screen_up` (default: +x button moves stage/piezo y, +y moves x).
  *Go to* has x, y, z boxes (stage in mm, piezo in um): fill any of them and click Go; empty boxes keep
  that axis where it is. Pasting all three (e.g. `[2.53, 1.18, 1.713]`) into one box fills all three. *Recenter piezo* moves the stage so the piezo is back mid-range without
  moving the spot. The same controls appear in the Auto Scan alignment step.
* **Home stage** (Move tab) and start-up checks:
  - If the stepper reads exactly (0, 0, 0) at start-up (freshly powered controller), the GUI homes it
    automatically and then returns to the last position it was at (last position with every coordinate
    above 0, remembered in `.last_stage_position.json`).
  - If only some coordinates read exactly 0, it shows an error: restart the stepper motors, then the GUI.
  - If any coordinate reads -0 (a tiny negative value, shown as -0.0000), it shows the same error
    without trying to home (homing does not work in that state).
  - If it reads negative coordinates (lost home position), it offers to home it.
  Homing goes z first (away from the objective), then x and y; the coordinates update live; it only says
  "Stage homed" after checking the stage really moves, and gives up after 60 s with a message to restart
  the stepper controller and relaunch the GUI.
* **Laser current sweep** (Laser tab): enter start / stop / step in % of max current and the counting time
  per setting; it sets each current, waits `sweep_settle_s` (0.3 s), and measures the total count rate
  (both detectors) at the current position, plotted live against %. The laser returns to its previous
  setting afterwards (also when stopped). *Save results...* writes NAME_image.png, NAME_data.csv (%, counts)
  and NAME_results.txt.
* **Laser**: the 637 nm laser (Thorlabs current driver over pyvisa). The % is **% of the diode
  controller's max current** (100 % = 0.14 A), not % of optical power. Its actual state is read at
  start-up, so if it is already on it shows as on and *Off* / *set current* work right away. 0 % = off.
* **Enter key**: typing a number and pressing Enter does the same as the box's button: laser % -> *Turn on
  / set current*, sweep settings -> *Start current sweep*, x / y from BL -> *Go there*, XY / Z scan
  settings -> *Start scan*, Go-to boxes -> *Go*.
* **Count Rate**: live count rate of detector 1, detector 2 and the sum, with a plot. Pause, clear,
  choose how much history to show, and save the data (CSV) and plot (PNG).
* **Alignment**: move to the bottom left marker using the camera, focus, *Save*; repeat for bottom right
  and top right. The system predicts top left from the other three and moves there; *Accept* it or adjust
  and *Save adjusted position*. It keeps cycling TL, BL, BR, TR until 4 in a row are accepted (any
  rectangle works). Every completed alignment is saved in `.alignment.json` (marker positions and the
  offsets from BL to each marker), so next time, at the BL step:
  - *New grid: I'm on its BL marker* places the other three markers using the saved layout (works for
    any grid with the same marker layout, also after restarting the GUI); then just accept / adjust.
  - *Re-check the saved markers* goes to the saved positions (same grid) for accept / adjust.
  *Restart alignment* asks whether to re-check the same markers, start a new grid from its BL marker, or
  restart fully by hand. *Go to a position on the sample* moves to x, y from the BL marker (default 75, 85
  um, along the marker edges), in focus. x = y = 0 is exactly the saved BL marker (position and focus); the
  directions of x / y and the focus tilt come from all four markers.
* **XY Scan**: centred on the current position, along the piezo x / y axes; the image axes are the piezo
  readouts (so a point can be typed straight into piezo Go-to). Scans up to 19.8 um work exactly like the
  old notebook's piezo scan (pyscan Sweep, loop0 = piezo x, loop1 = piezo y, dt = `dwell_s` = 0.1 s), so
  they give the same image: it goes to the first point and waits 3 s, hardware piezo x is the fast axis,
  every line starts again from the same side (no back-and-forth), and at each point it sets only the piezo
  axis that changes, waits 0.1 s (0.2 s at the start of a line) and takes one PicoHarp reading. z stays
  fixed; with the markers aligned and *Adjust z for the sample tilt* ticked, z follows the tilt of the
  marker plane (starting from the current focus), which is the only difference from the notebook.
  - Contrast: drag the cyan lines on the colour bar (top = brightest shown level, bottom = darkest), during
    or after the scan; *Auto contrast* or double-clicking the bar resets it. The saved image uses the
    contrast on screen.
  - Ranges up to 19.8 um (the piezo range) use the piezo only and the stepper never moves. The scan is
    centred on the current position where the piezo can reach it, shifted just enough to fit otherwise.
    The scan never covers more than the range typed in (e.g. 19.8 um at 2 um steps covers 18 um).
  - Larger ranges move the stepper during the scan, about every 19 um along each row (the piezo makes up
    the stepper's landing error, but sections can shift by up to ~1 um). The blue box warns about this.
  - When the scan finishes (or is stopped) the stepper and piezo go back to where the scan started (the
    piezo takes out the stepper's landing error). *Go to optimum* then moves to the brightest point (3 x 3
    centroid), using the piezo only if possible, and the tab shows the start and optimum positions.
    Double-click the image to go elsewhere.
* **Z Scan**: x, y fixed. *Sweep*: centred on the current z (current - range/2 to current + range/2,
  shifted to fit the piezo range if needed), or starting exactly at the current z going up (+z) or down
  (-z). Ranges up to 19.8 um use the piezo only. Afterwards it goes back to the start;
  *Go to optimum* moves to the best z (piezo first) and measures the count rate there (1 s average).
* **Count Rate** keeps reading while you are on other tabs (and uses the scan's own readings during scans).

  Every tab can be used in any order. Coordinates (readouts, Go-to boxes, saved files) are in the GUI
  frame: x and y are the axes the arrow buttons move (+x moves hardware y with the default config, so GUI
  x/y are swapped relative to Kinesis and the old notebook).

## Saved files

Nothing is saved automatically. After an XY or Z scan click *Save results...*, choose the folder to save
in and type a folder name, e.g. `G5_spot1`. The folder `G5_spot1` is created with:

* `G5_spot1_image.png`: the scan image or z plot (red line at the best z)
* `G5_spot1_data.csv` (opens in Excel): one row per point in the order measured: scan x / y
  (`x_marker_um, y_marker_um` or `x_piezo_um, y_piezo_um`), stage x/y/z (mm), piezo x/y/z (um),
  `counts_cps` (sum), `detector1_cps`, `detector2_cps`
* `G5_spot1_results.txt`: the start position (scan centre: stage and piezo, plus the position from the
  BL marker for marker scans), to repeat the scan, all scan settings, whether the stepper moved, and the
  optimum (where *Go to optimum* moved to, or the measured best point if you didn't go there)

The Count Rate tab's *Save data + plot...* works the same way (NAME_image.png = plot of the whole
recording, NAME_data.csv = every reading, NAME_results.txt = mean/std/min/max per detector, laser and
position).

## Counting

The PicoHarp reports count rates measured over fixed 100 ms gates. After every move the program waits one
full gate before using a reading, so each point only contains counts from its own position (reading
sooner mixes in the previous position: spots get smeared along the scan direction, their maxima drop, and
the "brightest point" shifts by up to one step). The dwell time (`dwell_s`, 0.1 s) is then averaged over
whole gates, so a longer dwell (e.g. 0.3 s = 3 gates) really reduces the noise. Each point therefore takes
about 0.1 s more than before; the time estimates include this.

After an XY scan the optimum is the centroid of the 3 x 3 pixels around the brightest one (at most half a
step from it), which stays reliable when a spot is only one pixel wide.

## Coordinates

The stage is in mm and the piezo in um. Internally everything uses one absolute frame in um:

    abs = 1000 * stage_mm + piezo_sign * piezo_um

Moves are done with the piezo when possible; when a target is outside the piezo range the stage moves
and the piezo makes up the difference using the stage **readback**, so stage inaccuracy does not matter.
Scans that fit inside the piezo range (e.g. fine scans, z sweeps) use the piezo only.

## Check these on the microscope before trusting it

Settings live in `confocal_gui/config.py`. To change them, create `confocal_config.json` next to
`run_confocal_gui.py` containing only the keys you want to change, e.g.
`{"piezo_sign": [1, -1, 1], "screen_right": ["y", -1]}`.

1. **`piezo_sign`** (most important). For each axis, a +1 um piezo move must move the spot the same
   way as a +1 um (0.001 mm) stage move. Test: focus on a marker edge, move the piezo +5 um in x and
   note where the marker went; move the piezo back and move the stage +5 um (use a 10 um stage step
   and compare half). If they go opposite ways, set that axis to -1. Repeat for y and z.
2. **`screen_right` / `screen_up`**: which hardware axis the +x and +y arrow buttons move (default:
   +x moves y, +y moves x). Labels never change. Each button moves only that one axis by the step size.
   The alignment and scans do not depend on this.
3. **`z_up_is_toward_objective`** and **`stage_limits_mm`**: the notebook says above 4 mm in z the
   sample hits the objective, so z is limited to 3.95 mm. Moves that lower z are done before xy moves.
4. **Laser**: check `max_current_A` in `lasers` matches what 100 % should mean.
5. **Time estimates** start at `overhead_per_point_s` (0.4 s per point on top of the dwell) and then
   learn from each finished scan (saved in `.scan_timing.json`; delete it to start over).

## Adding g2 later

The PicoHarp object is available as `hw.ph` in `PyscanHardware` (already in correlation mode); a g2
tab would be a fifth tab starting from the optimum found by Auto Scan.
