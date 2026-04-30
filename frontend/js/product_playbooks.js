/**
 * Product playbooks for repeatable ATLAS-0 beta loops.
 */

export const SAFETY_MISSIONS = [
  {
    id: 'cord-safari',
    title: 'Cable safari',
    audienceMode: 'pet',
    roomLabel: 'Cable safari',
    copy: 'Hunt for tempting cords, charger loops, and low wires that pets or feet can catch.',
    uploadHint: 'Record low shelves, desk corners, chargers, and floor paths where cords cross walking space.',
    steps: ['Start at floor height near outlets.', 'Pause on desks, TV stands, and charging corners.', 'Look for one cord you can route or tuck away.'],
  },
  {
    id: 'toddler-reach-test',
    title: 'Toddler reach test',
    audienceMode: 'toddler',
    roomLabel: 'Toddler reach test',
    copy: 'Scan everything below counter height and ask, "what could a small hand pull, climb, or tip?"',
    uploadHint: 'Keep low tables, shelves, handles, cords, and climbable furniture in frame for a steady moment.',
    steps: ['Walk the room from a lower angle.', 'Pause on reachable drawers, cords, and unstable furniture.', 'Fix or move one easy object after the report.'],
  },
  {
    id: 'renter-move-in-pass',
    title: 'Move-in receipt',
    audienceMode: 'renter',
    roomLabel: 'Move-in receipt',
    copy: 'Make a quick evidence-backed room note before moving furniture, boxes, or pets through the space.',
    uploadHint: 'Record walls, floors, corners, fixtures, and any existing hazards you may want documented.',
    steps: ['Scan each wall and corner slowly.', 'Capture fixtures, floor edges, and existing damage.', 'Download the PDF if the report finds anything worth saving.'],
  },
  {
    id: 'fall-zone-five',
    title: 'Five-minute fall zone',
    audienceMode: 'general',
    roomLabel: 'Fall zone',
    copy: 'Find one thing that could fall, tip, slide, or break if bumped during a busy day.',
    uploadHint: 'Record shelves, counters, tall furniture, and narrow walk paths where bumps are likely.',
    steps: ['Start with the tallest furniture.', 'Pause on shelves and counter edges.', 'Pick one quick stabilization fix.'],
  },
  {
    id: 'guest-ready-scan',
    title: 'Guest-ready sweep',
    audienceMode: 'general',
    roomLabel: 'Guest-ready sweep',
    copy: 'Before people come over, check the obvious trip paths and fragile surfaces once.',
    uploadHint: 'Record doorways, rug edges, coffee tables, walk paths, and anything fragile near elbows or bags.',
    steps: ['Walk the path a guest would take.', 'Pause on rugs, low tables, and doorways.', 'Remove one trip or bump hazard before guests arrive.'],
  },
];

export const CAPTURE_COACH_MODES = {
  general: {
    title: 'General room safety pass',
    promise: 'Best for a first beta scan when you just want a practical report.',
    route: ['Doorway and main walking path', 'Tall furniture and shelves', 'Tables, counters, and fragile edges', 'Floor-level trip points'],
    checks: ['Lights are on', 'Camera moves slowly', 'All four corners appear', 'Floor path appears', 'Shelves and counters pause in frame'],
    funPrompt: 'Try the "one quick fix" rule: pick one thing you can improve in under five minutes after the report.',
  },
  toddler: {
    title: 'Toddler reach pass',
    promise: 'Weights reachable, climbable, and pullable objects higher.',
    route: ['Low drawers and handles', 'Cords within reach', 'Climbable furniture', 'Heavy objects near edges'],
    checks: ['Camera drops to child height', 'Low surfaces stay visible', 'Cords and handles appear', 'Furniture bases appear', 'Sharp or fragile edges appear'],
    funPrompt: 'Scan from knee height for 20 seconds. It feels silly, but it changes what you notice.',
  },
  pet: {
    title: 'Pet safety sweep',
    promise: 'Highlights floor-level cords, tempting loops, unstable objects, and chewable clutter.',
    route: ['Outlet corners and charger nests', 'Desk and TV stand cables', 'Floor clutter and toys', 'Low shelves and plant stands'],
    checks: ['Camera starts low', 'Cord paths are visible', 'Under-table areas appear', 'Low shelves appear', 'Food/plant areas appear'],
    funPrompt: 'Play "what would a bored pet find first?" and let the scan catch the boring-but-real risks.',
  },
  renter: {
    title: 'Move-in receipt pass',
    promise: 'Keeps the report useful as evidence-backed notes before furniture or boxes change the room.',
    route: ['Each wall and corner', 'Floor edges and transitions', 'Fixtures and outlets', 'Existing scuffs or unsafe areas'],
    checks: ['Room label is specific', 'Walls and fixtures appear', 'Floor edges appear', 'Existing damage is captured', 'PDF export is planned'],
    funPrompt: 'Treat it like a tiny home inventory: one walkthrough now can save confusion later.',
  },
};

export const REPORT_DECISION_STEPS = [
  {
    id: 'fix',
    title: 'Fix one thing',
    copy: 'Start with the top action instead of trying to make the whole room perfect.',
  },
  {
    id: 'verify',
    title: 'Check the evidence',
    copy: 'Use frames and trust notes before acting on weaker findings.',
  },
  {
    id: 'rescan',
    title: 'Rescan the same room',
    copy: 'Reuse the room label so ATLAS-0 can compare the next pass.',
  },
  {
    id: 'share',
    title: 'Share the win',
    copy: 'Copy a clean summary or export the PDF when a fix is done.',
  },
];

export const BETA_SHARE_PROMPTS = [
  'I tried ATLAS-0: upload one room walkthrough, get a hazard report with evidence frames and practical fixes.',
  'Tiny room safety challenge: scan one room, fix one thing, rescan it. ATLAS-0 makes the before/after visible.',
  'ATLAS-0 is a room safety scan for renters, parents, pet owners, and anyone who wants a calmer home check.',
];
