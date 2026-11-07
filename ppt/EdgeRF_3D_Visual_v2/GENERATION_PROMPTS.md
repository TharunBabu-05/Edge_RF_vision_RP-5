# EdgeRF 3D visual presentation v2

Generated with the built-in image-generation tool. Original generated PNGs are preserved without retouching. The PDF embeds these images, one per page; text is rasterized, not editable PowerPoint text. Times New Roman-style serif lettering was requested; image generation does not embed font files.

These are concept illustrations of hardware, coverage and UI. Commercial pricing source: https://www.hamina.com/pricing (checked 4 November 2026). The INR amount uses an explicitly illustrative INR85/USD rate. Architecture is based on the local project code, with planned full-tilt orientation and calibrated VIO/SLAM clearly marked.

## Slide 1

Create ONE finished high-resolution landscape 16:9 presentation image, slide 1 of a coordinated three-slide engineering project pitch for "EdgeRF 3D". Not a photograph of a slide: the image IS the slide. Use case: productivity-visual / scientific infographic. A dramatically more visual, polished university engineering competition presentation. EXACTLY ONE slide in this image. Aim 3840x2160 if possible. All lettering Times New Roman / elegant readable serif; crisp, large, correct spelling, no sans-serif headings. White to very pale ice-blue background, midnight navy typography, cyan/royal blue and teal emphasis, amber and coral only for weak coverage. Consistent professional editorial hierarchy. Spacious, with large illustrated elements, NOT a generic grid of text boxes.

Top left small section label "01  PROBLEM STATEMENT"; top right "EdgeRF 3D".
Large title: "Wi-Fi is invisible. Coverage gaps are not."
Subtitle: "Affordable spatial surveys for IT offices, laboratories and institutions."

Hero occupying center/right ~60% of page: beautiful detailed isometric 3D cutaway of a compact contemporary office/laboratory building, three connected rooms with desks, computers, lab benches, interior walls. A recognizable wireless router emitting translucent curved blue radio wavefronts; colored floor heatmaps teal near router, amber through walls, coral in a distant meeting room. Subtle scan dots and an elegant floor grid. Make the surfaces sophisticated, luminous and dimensional, not cartoon clipart. Three clear anchored callouts, accurate serif lettering: "Office: strong signal", "Lab: uneven coverage", "Meeting room: weak zone". Small caption under hero "Illustrative coverage—not measured data."

Left ~34%: three concise issues, vertically stacked as elegant numbered editorial rows with generously sized distinct generated 3D pictograms (signal with blind spot, floor-map location pin, price tag + clock). Exact text:
"Hidden weak zones"
"Signal bars cannot explain room-to-room variation."
"Costly survey workflows"
"Professional tools, specialist skills and repeat visits add cost."
"Poor placement decisions"
"Without a map, access-point changes become guesswork."

Bottom horizontal band, carefully separated from hero: on left a money/rupee pictogram and compact cost proof:
"Commercial example: ₹83,300 / year"
"Hamina Planner: US$980/year at illustrative ₹85/US$."
On right a bright teal opportunity statement with a room-map icon:
"Our opportunity"
"A portable, lower-cost way to see where Wi-Fi is strong or weak."
Footer small but legible: "Vendor examples: Ekahau and Hamina—not an exhaustive list. Source: hamina.com/pricing, 4 Nov 2026. Taxes excluded; INR conversion is illustrative."
No company logos, no invented market statistics, no claim of only two competitors, no claim that every product costs over ₹60,000. No fake scientific charts. All text must be clean and readable with ample padding; don't add other text. The whole composition should feel dynamic and image-led, with the isometric room heatmap the dominant visual.

## Slide 2

Create ONE finished high-resolution 16:9 landscape presentation image: slide 2 of a three-slide EdgeRF 3D engineering pitch. The supplied image is a STYLE REFERENCE ONLY, not an edit target; create entirely new content and layout. Match its crisp Times New Roman-style serif typography, midnight navy lettering, white/ice-blue background, beautifully detailed isometric 3D illustrations, luminous teal/cyan, small warm accent colors, generous readable text, and consistent footer. EXACTLY ONE slide. No company logos. Prefer maximum resolution.

Top left section label "02  PROPOSED SOLUTION"; top right "EdgeRF 3D".
Main title "Walk the building. See Wi-Fi in 3D."
Subtitle "A portable Raspberry Pi 5 survey system with an interactive laptop view."

Main composition: a sweeping left-to-right visual workflow, with four large generated illustrations connected by broad, attractive, unambiguous arrows. Distinct connected stages, excellent whitespace and polished dimensional rendering. The illustrations occupy most of the slide, not small decorative thumbnails:
1. Left: portable Pi 5-style green single-board computer with USB ports, small camera, Wi-Fi antenna and a small IMU sensor board; stylized walking footsteps enter a small room grid. Caption "1  Capture" and smaller "Wi-Fi RSSI + camera frames + IMU".
2. Left-center: magnified luminous processor with three incoming blue/teal/violet data streams, feature-tracking points on an image plane and an orientation-axis glyph. Caption "2  Process on Pi 5" and smaller "Estimate movement and orientation".
3. Right-center: gorgeous cutaway multi-room building with teal/amber/coral signal heatmap, dotted survey path, room outlines and a few measurement pins. Caption "3  Build the spatial map" and smaller "Associate signal samples with positions".
4. Right: laptop showing a clean CONCEPTUAL 3D room heatmap, with understandable tiny network and room controls but no fake numerical results. Caption "4  Explore on the laptop" and smaller "Inspect coverage • Filter networks • Export".
Make these four illustration stages a single coherent journey and give each adequate width. Use real-looking generic hardware illustrations but avoid claiming an exact production enclosure or inventing an official board/logo.

Below this main pipeline, a strongly visual interaction vignette spanning the width: a tilted small Pi carrier and IMU with curved rotational arrows on left, a matching tilted 3D building view inside a laptop on right, connected by purple motion ribbon. Clear short heading "Move the device. Rotate the view."
Below heading precise status text "Current: heading rotation. Planned: full pitch-and-roll tilt."
Small secondary statement "The IMU controls viewing orientation; the camera estimates movement."
This is a proposed interaction illustration, not a measured latency claim.

Bottom three compact icon-supported benefit phrases, no paragraph boxes:
"Local edge processing" with small processor icon
"Room-aware surveys" with building/map-pin icon
"Accessible hardware" with small modular-parts icon
Footer: "Concept illustration • Full tilt and calibrated positioning are planned; accuracy and final hardware cost require validation."
All lettering must be legible serif, deliberate hierarchy, no overflowing text, no equations on this simple workflow slide, no fabricated performance metrics. Attractive enough to present directly as an image.

## Slide 3

Create ONE finished high-resolution landscape 16:9 presentation infographic image, slide 3 of an EdgeRF 3D engineering pitch. The supplied image is a visual STYLE REFERENCE only: match its elegant navy Times New Roman-style serif lettering, white/ice-blue canvas, beautifully detailed generated miniature 3D hardware/software illustrations, luminous cyan/teal and violet, and tasteful engineering-competition polish. This page must be a detailed, precise TECHNICAL ARCHITECTURE, more structured than the reference but equally visual. Exactly ONE slide. Highest resolution available. All text must be large enough to read, spelled accurately and not clipped. No logos or irrelevant decoration.

Top left: "03  TECHNICAL ARCHITECTURE"; top right "EdgeRF 3D".
Large title: "From sensor measurements to spatial Wi-Fi insight."
Small subtitle: "Raspberry Pi 5 edge processing • HTTP over local network • Three.js browser visualization"

Main architecture occupies the central 62% of slide. Three horizontal lanes with miniature high-quality isometric illustrations integrated into each node: BLUE for RF, TEAL for camera vision, VIOLET for IMU. Label the central processing domain "RASPBERRY PI 5" over an elegant subtle bounded area. Hardware on the left and a single large laptop/building graphic on right outside the Pi processing domain. Use continuous directional arrows attached to the correct nodes with no ambiguous crossings.

RF lane:
Wi-Fi adapter illustration, label "Wi-Fi scans", sublabel "iw / iwlist • RSSI + AP identity"
rightward arrow to radio-data list illustration, label "Acquire raw dBm"
rightward arrow to database/pin illustration SHARED with camera lane, label "Spatial RF samples", sublabel "(x, y, z, time, dBm)"
rightward arrow to floor-mesh/colored voxel illustration, label "Interpolate coverage", sublabel "IDW • Gaussian RBF • kernel methods"
rightward arrow to laptop output.

Camera lane below RF:
Camera illustration, label "Camera frames", sublabel "USB / V4L2"
rightward arrow to image feature points illustration, label "Visual odometry", sublabel "ORB → LK tracking → RANSAC", second small line "recoverPose → relative R, t"
rightward-and-up arrow that clearly joins the SAME spatial RF samples database. Under database put small italic "Monocular scale is uncalibrated."
Do NOT draw a second unrelated database.

IMU lane below camera, violet:
small sensor module illustration, label "6-axis IMU", sublabel "Accel + gyro • I²C"
rightward arrow to orientation-axis illustration, label "Orientation filter", sublabel "Gyro bias + complementary filter"
rightward arrow to angle/ribbon icon, label "Heading API", sublabel "80 ms polling • smoothed yaw"
rightward arrow to the laptop output, but separately marked "View rotation only". NO arrow from IMU to the visual-odometry or position database; sensor fusion is planned, not implemented.

Right output: large illustrated laptop with conceptual cutaway room heatmap, header "Three.js / WebGL"; labels below it "Rooms • RF heatmap • Walk path" and "SSID filters • CSV / PNG". A clear connection label on RF output arrow "Flask / JSON". Do not invent benchmark values.

Under main architecture: an experimental auxiliary pipeline with compact miniature 3D icons and visually connected teal arrows:
Heading "EXPERIMENTAL ROOM RECONSTRUCTION"
"4 room views" → "MiDaS depth + YOLOv8n" → "Pinhole + voxel reduction" → "RANSAC / SVD planes" → "Geometry JSON → /scene"
Small subcaption "ONNX Runtime • Approximate geometry; registration needs validation"
This is separate from the main live RF path, not falsely presented as existing calibrated fusion.

Near bottom, three compact math callouts in one organized row, precise serif typography:
"Normalize: I = clip((r + 90) / 60, 0, 1)"
"Display EMA: s(t) = 0.3 I(t) + 0.7 s(t−1)"
"IDW: r̂(p) = Σwᵢrᵢ / Σwᵢ; wᵢ = 1/dᵢ²"
Math captions can be slightly smaller, but still readable.

Bottom amber band, clearly labeled "PLANNED":
"Full pitch/roll tilt • Calibrated visual–inertial EKF / SLAM • Loop closure • Multi-floor height sampling"
Small footer "Concept hardware illustrations. Current architecture follows project code; planned methods are not yet integrated."

Use concise exact copy and powerful hardware/flow imagery. Avoid all-text grids, tiny text, thick black borders, excessive paragraphs, fake quantitative performance, fake Wi-Fi ray-tracing accuracy, and arrows implying existing IMU position fusion. Prefer only the specified labels. Clean, sophisticated, dynamic and presentation-ready.
