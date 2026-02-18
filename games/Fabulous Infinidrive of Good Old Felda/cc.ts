// ============================================================================= Constants and types

// Frames per second      // Frame width                      // Frame height
const FPS: number = 60;   const FRAME_WIDTH: number = 1280;   const FRAME_HEIGHT: number = 768;
// Half of the frame width          // Half of the frame height
const FRAME_WH = FRAME_WIDTH / 2;   const FRAME_HH = FRAME_HEIGHT / 2;

// ------------------------------------------------------------------------------------------- Pixel
interface Pixel {

    // X coordinate   // Y coordinate
    x: number;        y: number;

}

// ------------------------------------------------------------------------------------------- Point
interface Point {

    // X coordinate   // Y coordinate   // Z coordinate
    x: number;        y: number;        z: number;

}

// ---------------------------------------------------------------------------------------- Bodypart
interface Bodypart {

    // Indices of the points   // Color index
    indices: number[];         color: number;

}

// ----------------------------------------------------------------------------------------- Section
interface Section {

    // Ending index   // String with corresponding pattern
    i: number;        s: string;

}

// ---------------------------------------------------------------------------------------- Template
interface Template {

    // Length      // Color
    len: number;   col: number;
    // Either deltaX, or ease in length, deltaX, ease out length
    x: number[];
    // Either deltaY, or ease in length, deltaY, ease out length
    y: number[]
    // Either deltaR, or ease in length, deltaR, ease out length
    r: number[]
    // Lanes            // Obstacles            // Roof
    lanes: Section[];   obstacles: Section[];   roof: Section[];
    // Set of next connecting templates, one of them is chosen randomly
    next: number[];

}

// ========================================================================================= Objects

// ------------------------------------------------------------------------------------------ Player
class Player {

    // Geometry
    private static Geometry: Point[] = [
        { x: 15, y: 6.1, z: 31.2 }, { x: 15, y: 6.1, z: 27.7 }, // 0
        { x: 16, y: 10.6, z: 27.7}, { x: 16, y: 10.6, z: 31.2 },
        { x: 16, y: 12.3, z: 31.2 }, { x: 16, y: 12.3, z: 27.7 },
        { x: 16, y: 17, z: 26 }, { x: 15.7, y: 16.3, z: 30.5 },
        { x: 14.7, y: 19.2, z: 30 }, { x: 15, y: 24.7, z: 21.3 },
        { x: 12.7, y: 26, z: 21.7 }, { x: 12.5, y: 21.4, z: 30 }, // 10
        { x: -12.7, y: 26, z: 21.7 }, { x: -12.5, y: 21.4, z: 30 },
        { x: 0, y: 6.1, z: 27.7 }, { x: 0, y: 10.6, z: 27.7},
        { x: 0, y: 12.3, z: 27.7 }, { x: 0, y: 17, z: 26 },
        { x: 16, y: 17, z: 24.2 }, { x: 15, y: 24.7, z: 19.5 },
        { x: 15, y: 24.7, z: 6 }, { x: 16, y: 16, z: 6 }, // 20
        { x: 16, y: 14.5, z: 22.8 }, { x: 16, y: 14.5, z: 7 },
        { x: 16, y: 13, z: 22.6 }, { x: 16, y: 13, z: 7.3 },
        { x: 15, y: 6, z: 22.5 }, { x: 15, y: 6, z: 7.5 },
        { x: 16, y: 16, z: 2.5 }, { x: 15.8, y: 19, z: 2.5 },
        { x: 16, y: 16, z: 4.5 }, { x: 16, y: 15, z: 3.2 }, // 30
        { x: 16, y: 15, z: 1.5 }, { x: 16, y: 13.5, z: 2.7 },
        { x: 16, y: 13.5, z: 1 }, { x: 15.3, y: 8.5, z: 1.5 },
        { x: 15.3, y: 8, z: 2.5 }, { x: 10.4, y: 8.2, z: 0.5 },
        { x: 10.4, y: 13.5, z: 0 }, { x: 10.4, y: 15, z: 0 },
        { x: 10.4, y: 16, z: 0.7 }, { x: 10.4, y: 19, z: 0.7 }, // 40
        { x: 15.5, y: 21.5, z: 2.5 }, { x: 10.4, y: 22, z: 0.7 },
        { x: 10.5, y: 23.5, z: 0.7 }, { x: 15.3, y: 23, z: 2.5 },
        { x: 0, y: 6, z: 7.5 }, { x: 0, y: 13, z: 7.3 },
        { x: -15, y: 24.7, z: 21.3 }, { x: -15, y: 24.7, z: 6 },
        { x: -15.3, y: 23, z: 2.5 }, { x: -10.5, y: 23.5, z: 0.7 }, // 50
        { x: -10.4, y: 22, z: 0.7 }, { x: -15.5, y: 21.5, z: 2.5 },
        { x: -10.4, y: 19, z: 0.7 }, { x: -15.8, y: 19, z: 2.5 },
        { x: -16, y: 16, z: 2.5 }, { x: -10.4, y: 16, z: 0.7 },
        { x: -16, y: 15, z: 1.5 }, { x: -10.4, y: 15, z: 0 },
        { x: -16, y: 13.5, z: 1 }, { x: -10.4, y: 13.5, z: 0 }, // 60
        { x: -15.3, y: 8.5, z: 1.5 }, { x: -10.4, y: 8.2, z: 0.5 },
        { x: 10.9, y: 35.6, z: 17 }, { x: 9.3, y: 36.5, z: 18 },
        { x: -10.9, y: 35.6, z: 17 }, { x: -9.3, y: 36.5, z: 18 },
        { x: 15, y: 24.7, z: 14 }, { x: 15, y: 24.7, z: 12.5 },
        { x: 10.9, y: 35.6, z: 12 }, { x: 10.9, y: 35.6, z: 13.3 }, // 70
        { x: -15, y: 24.7, z: 14 }, { x: -15, y: 24.7, z: 12.5 },
        { x: -10.9, y: 35.6, z: 12 }, { x: -10.9, y: 35.6, z: 13.3 },
        { x: 14.6, y: 26.5, z: 5.5 }, { x: 14.5, y: 26.5, z: 2.5 },
        { x: 11.2, y: 26.5, z: 1 }, { x: -11.2, y: 26.5, z: 1 },
        { x: -14.5, y: 26.5, z: 2.5 }, { x: 14.2, y: 27.5, z: 5.5 }, // 80
        { x: 14.2, y: 27.5, z: 2.7 }, { x: 11.2, y: 27.5, z: 1.3 },
        { x: -14.2, y: 27.5, z: 2.7 }, { x: -11.2, y: 27.5, z: 1.3 },
        { x: 10.9, y: 35.6, z: 8 }, { x: 10, y: 37.5, z: 6.5 },
        { x: 10, y: 38, z: 16.2 }, { x: -10.9, y: 35.6, z: 8 },
        { x: -10, y: 37.5, z: 6.5 }, { x: -10, y: 38, z: 16.2 }, // 90
        { x: 8.5, y: 36.5, z: 5.5 }, { x: -8.5, y: 36.5, z: 5.5 },
        { x: 0.8, y: 26.2, z: 0.7 }, { x: 0.8, y: 24.2, z: 0.7 },
        { x: -0.8, y: 26.2, z: 0.7 }, { x: -0.8, y: 24.2, z: 0.7 },
        { x: 5, y: 22, z: 0.7 }, { x: 5, y: 19, z: 0.7 },
        { x: -5, y: 22, z: 0.7 }, { x: -5, y: 19, z: 0.7 }, // 100
        { x: 15.8, y: 20.4, z: 15.5 }, { x: 15.7, y: 21.4, z: 15.5 },
        { x: 15.7, y: 21.4, z: 14.3 }, { x: 15.8, y: 20.4, z: 14.3 },
        { x: 15.7, y: 21.8, z: 7 }, { x: 15.8, y: 20.8, z: 7 },
        { x: 15.8, y: 20.8, z: 6.1 }, { x: 15.7, y: 21.8, z: 6.1 },
        { x: 15.3, y: 5, z: 27.2 }, { x: 15.3, y: 11, z: 27.2 }, // 110
        { x: 15.3, y: 15.3, z: 26 }, { x: 15.3, y: 0.9, z: 26 },
        { x: 15.3, y: 15.9, z: 25 }, { x: 15.3, y: 0.3, z: 25 },
        { x: 8, y: 15.8, z: 25 }, { x: 12, y: 0.2, z: 25 },
        { x: 15.3, y: 5, z: 7.2 }, { x: 15.3, y: 11, z: 7.2 },
        { x: 15.3, y: 15.3, z: 6 }, { x: 15.3, y: 0.9, z: 6 }, // 120
        { x: 15.3, y: 15.9, z: 5 }, { x: 15.3, y: 0.3, z: 5 },
        { x: 8, y: 15.8, z: 5 }, { x: 12, y: 0.2, z: 5 },
        { x: -15.3, y: 5, z: 7.2 }, { x: -15.3, y: 11, z: 7.2 },
        { x: -15.3, y: 15.3, z: 6 }, { x: -15.3, y: 0.9, z: 6 },
        { x: -15.3, y: 15.9, z: 5 }, { x: -15.3, y: 0.3, z: 5 }, // 130
        { x: -8, y: 15.8, z: 5 }, { x: -12, y: 0.2, z: 5 },
    ];

    // Bodyparts
    private static Body: Bodypart[] = [
        { indices: [0, 1, 2, 3], color: 3 }, // Front lower fender
        { indices: [2, 3, 4, 5], color: 2 }, // Front fender trim
        { indices: [4, 5, 6, 7], color: 4 }, // Front upper fender
        { indices: [6, 7, 8, 9], color: 4 }, // Lower hood
        { indices: [10, 11, 13, 12], color: 6 }, // Hood
        { indices: [8, 9, 10, 11], color: 5 }, // Side hood
        { indices: [1, 2, 15, 14], color: 1 }, // Front lower fender cover
        { indices: [2, 5, 16, 15], color: 1 }, // Front fender trim cover
        { indices: [5, 6, 17, 16], color: 1 }, // Front upper fender cover
        { indices: [110, 111, 112, 113], color: 2 }, // Tire
        { indices: [112, 114, 115, 113], color: 2 }, // Tire
        { indices: [114, 116, 117, 115], color: 2 }, // Tire
        { indices: [10, 9, 48, 12], color: 2}, // Dashboard
        { indices: [10, 9, 64, 65], color: 5}, // Left front window
        { indices: [48, 12, 67, 66], color: 5 }, // Right front window
        { indices: [6, 9, 19, 18], color: 4 }, // Between hood and doors
        { indices: [26, 24, 25, 27], color: 3 }, // Lower doors
        { indices: [27, 25, 47, 46], color: 1 }, // Lower doors cover
        { indices: [24, 22, 23, 25], color: 2 }, // Doors trim
        { indices: [23, 25, 47, 47], color: 1 }, // Doors trim cover
        { indices: [22, 18, 21, 23], color: 4 }, // Side doors
        { indices: [23, 21, 47, 47], color: 1 }, // Side doors cover
        { indices: [118, 119, 120, 121], color: 2 }, // Tire
        { indices: [120, 122, 123, 121], color: 2 }, // Tire
        { indices: [122, 124, 125, 123], color: 2 }, // Tire
        { indices: [126, 127, 128, 129], color: 2 }, // Tire
        { indices: [128, 130, 131, 129], color: 2 }, // Tire
        { indices: [130, 132, 133, 131], color: 2 }, // Tire
        { indices: [18, 19, 20, 21], color: 4 }, // Upper doors
        { indices: [102, 103, 104, 105], color: 2 }, // Door handle
        { indices: [106, 107, 108, 109], color: 2 }, // Door handle
        { indices: [9, 48, 49, 20], color: 1}, // Interior cover
        { indices: [68, 69, 70, 71], color: 5 }, // Left middle window
        { indices: [72, 73, 74, 75], color: 5 }, // Right middle window
        { indices: [20, 21, 28, 45], color: 4 }, // Between doors and rear fender
        { indices: [28, 30, 31, 32], color: 4 }, // Side upper rear fender
        { indices: [31, 32, 34, 33], color: 2 }, // Side rear fender trim
        { indices: [33, 34, 35, 36], color: 3 }, // Side lower rear fender
        { indices: [34, 35, 37, 38], color: 3 }, // Left rear lower fender
        { indices: [61, 60, 62, 63], color: 3 }, // Right rear lower fender
        { indices: [37, 38, 61, 63], color: 3 }, // Rear lower fender
        { indices: [32, 34, 38, 39], color: 2 }, // Left rear fender trim
        { indices: [59, 58, 60, 61], color: 2 }, // Right rear fender trim
        { indices: [38, 39, 59, 61], color: 2 }, // Rear fender trim
        { indices: [28, 32, 39, 40], color: 6 }, // Left rear upper fender
        { indices: [57, 56, 58, 59], color: 6 }, // Right rear upper fender
        { indices: [39, 40, 57, 59], color: 6 }, // Rear upper fender
        { indices: [29, 28, 40, 41], color: 4 }, // Left lower back
        { indices: [54, 55, 56, 57], color: 4 }, // Right lower back
        { indices: [29, 41, 43, 42], color: 0 }, // Left lower light
        { indices: [53, 52, 54, 55], color: 0 }, // Right lower light
        { indices: [43, 40, 57, 52], color: 4 }, // Lower back
        { indices: [42, 43, 44, 45], color: 2 }, // Left light trim
        { indices: [50, 51, 52, 53], color: 2 }, // Right light trim
        { indices: [43, 44, 51, 52], color: 2 }, // Light trim
        { indices: [64, 65, 65, 88], color: 5 }, // Roof and front window filler
        { indices: [49, 20, 45, 50], color: 1}, // Interior cover
        { indices: [86, 64, 66, 89], color: 2 }, // Roof cover
        { indices: [86, 92, 93, 89], color: 2 }, // Roof cover
        { indices: [20, 45, 77, 76], color: 4 }, // Lower rear side window
        { indices: [45, 77, 78, 44], color: 0 }, // Left upper light
        { indices: [80, 50, 51, 79], color: 0 }, // Right upper light
        { indices: [44, 78, 79, 51], color: 4 }, // Upper back
        { indices: [77, 76, 81, 82], color: 5 }, // Rear side lower window
        { indices: [82, 83, 78, 77], color: 5 }, // Left rear lower window
        { indices: [84, 85, 79, 80], color: 5 }, // Right rear lower window
        { indices: [85, 83, 78, 79], color: 5 }, // Rear lower window
        { indices: [86, 87, 88, 64], color: 5 }, // Left roof
        { indices: [89, 90, 91, 66], color: 5 }, // Right roof
        { indices: [82, 81, 86, 87], color: 5 }, // Rear side window
        { indices: [82, 83, 92, 87], color: 5 }, // Left rear window
        { indices: [84, 85, 93, 90], color: 5 }, // Right rear window
        { indices: [92, 93, 90, 87], color: 5 }, // Rear window
        { indices: [90, 87, 88, 91], color: 6 }, // Roof
        { indices: [94, 95, 97, 96], color: 2 }, // Skoda logo
        { indices: [98, 99, 101, 100], color: 5 }, // SPZ
    ];

    // X coordinate                // Z coordinate                // Distance traveled
    public static x: number = 0;   public static z: number = 0;   public static distance: number = 1;

    // Delta X                          // Max steering                   // Steering acceleration
    public static deltaX: number = 0;   public static maxX: number = 3;   public static steer: number = 0.15;

    // Projected geometry, deep copy of Geometry using JSON serialization... just JavaScript things
    private static Projected: Point[] = JSON.parse(JSON.stringify(Player.Geometry));

    // Speed                           // Base Z coordinate
    public static speed: number = 0;   public static baseZ: number = 38;

    // Keyboard
    private static Keyboard = class {

        // Steer left                          // Steer right
        public static left: boolean = false;   public static right: boolean = false;
        // Most recent touch identifier
        public static touch: number = 0;

    }

    // Camera
    public static Camera = class {

        // Field of view                    // Height of the camera above the road
        private static fov: number = 100;   public static height: number = 50;
        // Distance to projection plane
        public static distance: number = 1 / Math.tan((this.fov / 2) * Math.PI / 180);

    }

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  Initialize, Update, Render

    // Initialize
    public static initialize(): void {
        document.addEventListener('keydown', function(event: KeyboardEvent): void {
            if (!Game.running) Game.start();
            else {
                if (event.key == 'ArrowLeft' || event.key == 'a') Player.Keyboard.left = true;
                else if (event.key == 'ArrowRight' || event.key == 'd') Player.Keyboard.right = true;
                else if (event.key == 'r' && Game.counter > 620) {
                    Game.restart();
                }
            }
        });
        document.addEventListener('keyup', function(event: KeyboardEvent): void {
            if (event.key == 'ArrowLeft' || event.key == 'a') Player.Keyboard.left = false;
            else if (event.key == 'ArrowRight' || event.key == 'd') Player.Keyboard.right = false;
        });
        Page.canvas.addEventListener('mousedown', function(event: MouseEvent): void {
            if (!Game.running) Game.start();
            else {
                if (event.clientX - Page.canvas.getBoundingClientRect().left <
                    Page.canvas.getBoundingClientRect().width / 2) Player.Keyboard.left = true;
                else Player.Keyboard.right = true;
            }
        });
        Page.canvas.addEventListener('mouseup', function(event: MouseEvent): void {
            Player.Keyboard.left = false;
            Player.Keyboard.right = false;
        });
        Page.canvas.addEventListener('touchstart', function(event: TouchEvent): void {
            if (!Game.running) Game.start();
            else if (Game.counter > 620) Game.restart();
            else {
                if (event.changedTouches[event.changedTouches.length - 1].clientX -
                    Page.canvas.getBoundingClientRect().left <
                    Page.canvas.getBoundingClientRect().width / 2) {
                    Player.Keyboard.left = true;
                    Player.Keyboard.right = false;
                } else {
                    Player.Keyboard.right = true;
                    Player.Keyboard.left = false;
                }
            }
        });
        Page.canvas.addEventListener('touchend', function(event: TouchEvent): void {
            if (event.touches.length == 0) {
                Player.Keyboard.left = false;
                Player.Keyboard.right = false;
            }
        });
    }

    // Update
    public static update(): void {
        // Steering
        if (Player.Keyboard.left) {
            Player.deltaX -= Player.steer;
            if (Player.deltaX < -Player.maxX)
                Player.deltaX = -Player.maxX;
        } else if (Player.Keyboard.right) {
            Player.deltaX += Player.steer;
            if (Player.deltaX > Player.maxX)
                Player.deltaX = Player.maxX;
        } else {
            Player.deltaX *= 0.95;
        }
        if (Game.paused) Player.deltaX = 0;
        Player.x += Player.deltaX;

        // Distance
        Player.z += Player.speed;
        if (Player.z > Road.Segments.depth) {
            Player.z -= Road.Segments.depth;
            Player.distance++;
            Generator.generate(Player.distance);
        }
    }

    // Render
    public static render(): void {
        Player.project();

        for (let part in Player.Body) {
            if (Player.x < 0)
                Utility.polygon(
                    { x: Math.round(FRAME_WH + Player.Projected[Player.Body[part].indices[0]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[0]].y) },
                    { x: Math.round(FRAME_WH + Player.Projected[Player.Body[part].indices[1]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[1]].y) },
                    { x: Math.round(FRAME_WH + Player.Projected[Player.Body[part].indices[2]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[2]].y) },
                    { x: Math.round(FRAME_WH + Player.Projected[Player.Body[part].indices[3]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[3]].y) },
                      Page.Colors.car[Player.Body[part].color]);
            else
                Utility.polygon(
                    { x: Math.round(FRAME_WH - Player.Projected[Player.Body[part].indices[0]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[0]].y) },
                    { x: Math.round(FRAME_WH - Player.Projected[Player.Body[part].indices[1]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[1]].y) },
                    { x: Math.round(FRAME_WH - Player.Projected[Player.Body[part].indices[2]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[2]].y) },
                    { x: Math.round(FRAME_WH - Player.Projected[Player.Body[part].indices[3]].x),
                      y: Math.round(Player.Projected[Player.Body[part].indices[3]].y) },
                      Page.Colors.car[Player.Body[part].color]);
        }
    }

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   Private

    // Project a bodypart onto the projection plane
    private static project(): void {
        let incline: number = Road.Segments.array[(Player.distance - 1) % Road.Segments.count].deltaY * 10
            + Road.Segments.array[(Player.distance - 1) % Road.Segments.count].deltaR * Player.x * 1.5;

        if (Game.paused) {
            if (Game.counter >= 120) {
                if (Game.counter < 130)
                    incline += Utility.easeInQuint(0, 6, (Game.counter - 120) / 10);
                else if (Game.counter < 170)
                    incline += Utility.easeInQuint(6, 0, (Game.counter - 130) / 40);
            }
        }

        for (let i: number = 0; i < Player.Geometry.length; i++) {
            Player.Projected[i].x = Player.Geometry[i].x * 0.8 - Math.abs(Player.x) / 6;
            Player.Projected[i].y = Player.Geometry[i].y + incline * (Player.Geometry[i].z - 5) * 0.05
                - Player.Camera.height;
            Player.Projected[i].z = Player.Geometry[i].z + Player.baseZ;
            let scale: number = Player.Camera.distance / Player.Projected[i].z;
            Player.Projected[i].x = Player.Projected[i].x * scale * FRAME_WH;
            Player.Projected[i].y = (1 - Player.Projected[i].y * scale) * (FRAME_HH);
        }
    }

}

// ----------------------------------------------------------------------------------------- Segment
class Segment {

    // Delta of X coordinate of the back edge   // Delta of Y coordinate of the back edge
    public deltaX: number = 0;                  public deltaY: number = 0;
    // Delta of rotation         // Color index
    public deltaR: number = 0;   public color: number = 0;
    // Lanes: [ and ] mark edges, | marks continous line, . marks dashed line, space marks empty
    public lanes: string = '[    ]';
    // Obstacles: . o O mark connecting obstacles of increasing size,
    // , q Q mark split obstacles of increasing size, space marks empty
    public obstacles: string = '      ';
    // Roof: # marks roof, space marks empty
    public roof: string = '     ';

}

// -------------------------------------------------------------------------------------------- Road
class Road {

    // Offset of the road from the center of the screen for curves and hills
    private static Offset = class {

        // X coordinate                // Y coordinate
        public static x: number = 0;   public static y: number = 0;
        // Delta of X coordinate            // Delta of Y coordinate
        public static deltaX: number = 0;   public static deltaY: number = 0;
        // Delta of rotation
        public static deltaR: number = 0;
        // Sine of delta rotation            // Cosine of delta rotation
        public static sin: number = 0;   public static cos: number = 1;

        // Increment the offset
        public static increment(id: number) {
            Road.Offset.x += Road.Offset.deltaX;
            Road.Offset.y += Road.Offset.deltaY;
            Road.Offset.sin = Math.sin(Road.Offset.deltaR);
            Road.Offset.cos = Math.cos(Road.Offset.deltaR);
            Road.Offset.deltaX += Road.Offset.cos * Road.Segments.array[id].deltaX
                - Road.Offset.sin * Road.Segments.array[id].deltaY;
            Road.Offset.deltaY += Road.Offset.cos * Road.Segments.array[id].deltaY
                + Road.Offset.sin * Road.Segments.array[id].deltaX;
            Road.Offset.deltaR += Road.Segments.array[id].deltaR;
        }

        // Decrement the offset
        public static decrement(id: number) {
            Road.Offset.deltaR -= Road.Segments.array[id].deltaR;
            Road.Offset.sin = Math.sin(Road.Offset.deltaR);
            Road.Offset.cos = Math.cos(Road.Offset.deltaR);
            Road.Offset.deltaX -= Road.Offset.cos * Road.Segments.array[id].deltaX
                - Road.Offset.sin * Road.Segments.array[id].deltaY;
            Road.Offset.deltaY -= Road.Offset.cos * Road.Segments.array[id].deltaY
                + Road.Offset.sin * Road.Segments.array[id].deltaX;
            Road.Offset.x -= Road.Offset.deltaX;
            Road.Offset.y -= Road.Offset.deltaY;
        }

    }

    // Front edge of the last projected road segment
    private static Front = class {

        // X coordinate                // Y coordinate                // Z coordinate
        public static x: number = 0;   public static y: number = 0;   public static z: number = 0;
        // Scale                           // Width
        public static scale: number = 0;   public static width: number = 0;
        // Curb width                     // Lane stripe width
        public static curb: number = 0;   public static stripe: number = 0;
        // Sine of delta rotation        // Cosine of delta rotation
        public static sin: number = 0;   public static cos: number = 1;

    }

    // Back edge of the last projected road segment
    private static Back: typeof Road.Front = Object.create(Road.Front);

    // Segments
    public static Segments = class {
    
        // Number of segments               // Length of each segment
        public static count: number = 128;   public static depth: number = 40;
        // Array of segments
        public static array: Segment[];

    }

    // Offsets of the holes in the road for collision detection
    public static Holes: number[] = [-217, -127, -37, 53, 143, -143, -53, 37, 127, 217];
    // Offsets of the obstacles on the road for collision detection
    public static Obstacles: number[] = [-206, -116, -26, 64, 153, 243];

    // Width of the road
    public static width: number = 30;

    // Backups for the correct rendering of the car behind some obstacles
    private static backup1: typeof Road.Front = Object.create(Road.Front);
    private static backup2: typeof Road.Back = Object.create(Road.Back);
    private static backup3: typeof Road.Offset = Object.create(Road.Offset);

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  Initialize, Update, Render

    // Initialize
    public static initialize(): void {
        Road.Segments.array = new Array(Road.Segments.count);
        for (let i: number = 0; i < Road.Segments.count; i++) {
            Road.Segments.array[i] = new Segment();
            Road.Segments.array[i].deltaX = 0;
            Road.Segments.array[i].deltaY = 0;
            Road.Segments.array[i].deltaR = 0;
            Road.Segments.array[i].color = i % 16 < 8 ? 0 : 1;
            Road.Segments.array[i].lanes = '[....]';
            Road.Segments.array[i].obstacles = '      ';
            Road.Segments.array[i].roof = '     ';
        }
    }

    // Render
    public static render(): void {
        let id: number = (Player.distance - 1) % Road.Segments.count;
        Road.Offset.x = 0;
        Road.Offset.y = 0;
        // Smoothly interpolate the offset of the road from the center of the screen
        // The road warps after each passed segment without this
        Road.Offset.deltaX = Road.Segments.array[id].deltaX * (1 - Player.z / Road.Segments.depth);
        Road.Offset.deltaY = Road.Segments.array[id].deltaY * (1 - Player.z / Road.Segments.depth);
        Road.Offset.deltaR = Road.Segments.array[id].deltaR * (1 - Player.z / Road.Segments.depth);

        // Compute the offset of the road for the very last segment
        for (let i: number = 0; i < Road.Segments.count; i++) {
            id = (Player.distance + i) % Road.Segments.count;
            Road.Offset.increment(id);
        }

        // Compute the offset of the very last edge of the road
        Road.Front.x = Road.Offset.x + Road.Offset.deltaX - Player.x / 2;
        Road.Front.y = Road.Offset.y + Road.Offset.deltaY - Player.Camera.height;
        Road.Front.z = Road.Segments.depth * (Road.Segments.count + 3) - Player.z;
        Road.Front.scale = Player.Camera.distance / Road.Front.z;
        Road.Front.x = (1 + Road.Front.x * Road.Front.scale) * (FRAME_WH);
        Road.Front.y = (1 - Road.Front.y * Road.Front.scale) * (FRAME_HH);
        Road.Front.width = Road.width * Road.Front.scale * (FRAME_WH);
        Road.Front.sin = Road.Offset.sin;
        Road.Front.cos = Road.Offset.cos;

        // Render the road segments starting from the very last
        for (let i: number = Road.Segments.count - 1; i > -1; i--) {
            Road.swap();
            id = (Road.Segments.count + Player.distance + i) % Road.Segments.count;
            Road.project();
            Road.segment(id);
            let yFront: number = 64 * FRAME_HEIGHT * (Player.Camera.distance / (Road.Front.z));
            let yBack: number = 64 * FRAME_HEIGHT * (Player.Camera.distance / (Road.Back.z));
            if (yFront <= yBack)
                Road.roof(id, yFront, yBack);
            Road.obstacles(id, false);
            if (yBack < yFront)
                Road.roof(id, yFront, yBack);
            Road.Offset.decrement(id);
        }

        // Very hacky way to render the car behind some obstacles
        // Create a backup of the front and back edges of the road and the offset
        // Render everything
        // Restore the backup
        // Render those obstacles which are in front of the car
        Road.backup1.x = Road.Front.x;
        Road.backup1.y = Road.Front.y;
        Road.backup1.z = Road.Front.z;
        Road.backup1.scale = Road.Front.scale;
        Road.backup1.width = Road.Front.width;
        Road.backup1.curb = Road.Front.curb;
        Road.backup1.stripe = Road.Front.stripe;
        Road.backup1.sin = Road.Front.sin;
        Road.backup1.cos = Road.Front.cos;
        Road.backup2.x = Road.Back.x;
        Road.backup2.y = Road.Back.y;
        Road.backup2.z = Road.Back.z;
        Road.backup2.scale = Road.Back.scale;
        Road.backup2.width = Road.Back.width;
        Road.backup2.curb = Road.Back.curb;
        Road.backup2.stripe = Road.Back.stripe;
        Road.backup2.sin = Road.Back.sin;
        Road.backup2.cos = Road.Back.cos;
        Road.backup3.x = Road.Offset.x;
        Road.backup3.y = Road.Offset.y;
        Road.backup3.deltaX = Road.Offset.deltaX;
        Road.backup3.deltaY = Road.Offset.deltaY;
        Road.backup3.deltaR = Road.Offset.deltaR;
        Road.backup3.sin = Road.Offset.sin;
        Road.backup3.cos = Road.Offset.cos;
        // Dear readers, be kind to me, please close your eyes and pretend this abomination does not exist

        for (let i: number = -1; i > -3; i--) {
            Road.swap();
            id = (Road.Segments.count + Player.distance + i) % Road.Segments.count;
            Road.project();
            Road.segment(id);
            let yFront: number = 64 * FRAME_HEIGHT * (Player.Camera.distance / (Road.Front.z));
            let yBack: number = 64 * FRAME_HEIGHT * (Player.Camera.distance / (Road.Back.z));
            if (yFront <= yBack)
                Road.roof(id, yFront, yBack);
            Road.obstacles(id, false);
            if (yBack < yFront)
                Road.roof(id, yFront, yBack);
            Road.Offset.decrement(id);
        }

        if (Game.counter < 429)
            Player.render();

        // Oh no, it's back!
        Road.Front.x = Road.backup1.x;
        Road.Front.y = Road.backup1.y;
        Road.Front.z = Road.backup1.z;
        Road.Front.scale = Road.backup1.scale;
        Road.Front.width = Road.backup1.width;
        Road.Front.curb = Road.backup1.curb;
        Road.Front.stripe = Road.backup1.stripe;
        Road.Front.sin = Road.backup1.sin;
        Road.Front.cos = Road.backup1.cos;
        Road.Back.x = Road.backup2.x;
        Road.Back.y = Road.backup2.y;
        Road.Back.z = Road.backup2.z;
        Road.Back.scale = Road.backup2.scale;
        Road.Back.width = Road.backup2.width;
        Road.Back.curb = Road.backup2.curb;
        Road.Back.stripe = Road.backup2.stripe;
        Road.Back.sin = Road.backup2.sin;
        Road.Back.cos = Road.backup2.cos;
        Road.Offset.x = Road.backup3.x;
        Road.Offset.y = Road.backup3.y;
        Road.Offset.deltaX = Road.backup3.deltaX;
        Road.Offset.deltaY = Road.backup3.deltaY;
        Road.Offset.deltaR = Road.backup3.deltaR;
        Road.Offset.sin = Road.backup3.sin;
        Road.Offset.cos = Road.backup3.cos;

        for (let i: number = -1; i > -3; i--) {
            Road.swap();
            id = (Road.Segments.count + Player.distance + i) % Road.Segments.count;
            Road.project();
            Road.obstacles(id, true);
            Road.Offset.decrement(id);
        }
    }

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   Private

    // Copy the front edge of the further segment to the back edge of the nearer segment
    private static swap(): void {
        Road.Back.x = Road.Front.x;
        Road.Back.y = Road.Front.y;
        Road.Back.z = Road.Front.z;
        Road.Back.scale = Road.Front.scale;
        Road.Back.width = Road.Front.width;
        Road.Back.curb = Road.Front.curb;
        Road.Back.stripe = Road.Front.stripe;
        Road.Back.sin = Road.Front.sin;
        Road.Back.cos = Road.Front.cos;
    }

    // Project a road segment onto the projection plane
    private static project(): void {
        Road.Front.x = Road.Offset.x - Player.x / 2;
        Road.Front.y = Road.Offset.y - Player.Camera.height;
        Road.Front.z = Road.Back.z - Road.Segments.depth;
        Road.Front.scale = Player.Camera.distance / Road.Front.z;
        Road.Front.x = (1 + Road.Front.x * Road.Front.scale) * (FRAME_WH);
        Road.Front.y = (1 - Road.Front.y * Road.Front.scale) * (FRAME_HH);
        Road.Front.width = Road.width * Road.Front.scale * (FRAME_WH);
        Road.Front.sin = Road.Offset.sin;
        Road.Front.cos = Road.Offset.cos;
    }

    // Render a road segment
    private static segment(id: number): void {
        let frontW: number = Road.Front.width;
        let backW: number = Road.Back.width;
        let frontCurb: number = frontW * 0.15;
        let backCurb: number = backW * 0.15;
        let frontStrip: number = frontW * 0.03;
        let backStrip: number = backW * 0.03;

        let w: number;

        // Road
        for (let i: number = 0; i < 6; i++) {
            // Check if the current lane has a road
            if (Road.Segments.array[id].lanes[i] == '[')
                w = i * 2 - 4;
            else if (Road.Segments.array[id].lanes[i] == ']') {
                let v: number = (i - 1) * 2 - 4;
                Road.polygon(Road.Front.width, Road.Back.width, w, v,
                             Page.Colors.road[Road.Segments.array[id].color]);
            }
        }

        // Lane stripes
        for (let i: number = 0; i < 6; i++) {
            let lane: string = Road.Segments.array[id].lanes[i];
            w = i * 2 - 5;

            // Curbs on the sides of the road
            if (lane == '[' || lane == ']')
                Road.polygon(frontCurb, backCurb, w, w, 'white');
            // Continuous lane line
            else if (lane == '|')
                Road.polygon(frontStrip, backStrip, w, w, 'white');
            // Dashed lane line
            else if (lane == '.')
                if (id % 4 < 1)
                    Road.polygon(frontStrip, backStrip, w, w, 'white');
        }
    }

    // Obstacles
    private static obstacles(id: number, magic: boolean): void {
        let carX: number = 0;

        // Magic boolean, awesome right?
        if (magic) for (let i: number = 0; i < 5; i++) if (Player.x > Road.Holes[i]) carX = i;

        for (let i: number = 0; i < 6; i++) {
            let obstacle: string = Road.Segments.array[id].obstacles[i];

            let y: number;
            if (obstacle == ',' || obstacle == '.') y = 12;
            else if (obstacle == 'q' || obstacle == 'o') y = 24;
            else if (obstacle == 'Q' || obstacle == 'O') y = 64;

            // Check if an obstacle changed from the previous road segment
            let match: boolean = true;
            if (Road.Segments.array[(id + Road.Segments.count - 1) %
                Road.Segments.count].obstacles[i] != obstacle)
                match = false;

            // Obstacles which do not connect when on adjacent lines
            if (obstacle == ',' || obstacle == 'q' || obstacle == 'Q') {

                let u: number = Road.Front.width * 0.03;
                let v: number = Road.Back.width * 0.03;
                let w: number = 5 - i * 2;

                let yFront: number = y * FRAME_HEIGHT * (Player.Camera.distance / (Road.Front.z));
                let yBack: number = y * FRAME_HEIGHT * (Player.Camera.distance / (Road.Back.z));

                // Check if the obstacle is in front of the car
                if (magic) {
                    if (carX == 2) continue;
                    else if (carX > 2 && carX < i) continue;
                    else if (carX < 2 && carX > i - 1) continue;
                }

                // Side
                Utility.polygon(
                    { x: Road.Front.x - w * Road.Front.width,
                      y: Road.Front.y },
                    { x: Road.Front.x - w * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x - w * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - w * Road.Back.width,
                      y: Road.Back.y },
                      Page.Colors.obstacle[1]);
                // Top
                Utility.polygon(
                    { x: Road.Front.x - u - w * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Front.x + u - w * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x + v - w * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - v - w * Road.Back.width,
                      y: Road.Back.y - yBack},
                      Page.Colors.obstacle[0]);

                if (!match)
                    // Front
                    Utility.polygon(
                        { x: Road.Front.x - u - w * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x + u - w * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x + u - w * Road.Front.width,
                          y: Road.Front.y - yFront },
                        { x: Road.Front.x - u - w * Road.Front.width,
                          y: Road.Front.y - yFront},
                          Page.Colors.obstacle[0]);

            // Obstacles which connect when on adjacent lines
            } else if (obstacle == '.' || obstacle == 'o' || obstacle == 'O') {
                let prev: [string, number] = [obstacle, i];
                while (Road.Segments.array[id].obstacles[i] == prev[0]) {
                    i++;
                }
                i--;

                let u: number = Road.Front.width * 0.03;
                let v: number = Road.Back.width * 0.03;
                let wLeft = 5 - prev[1] * 2;
                let wRight = 5 - i * 2;

                let yFront: number = y * 720 * (Player.Camera.distance / (Road.Front.z));
                let yBack: number = y * 720 * (Player.Camera.distance / (Road.Back.z));

                // Check if the obstacle is in front of the car
                if (magic) {
                    if (carX == 2) continue;
                    else if (carX > 2 && carX < prev[1]) continue;
                    else if (carX < 2 && carX > i - 1) continue;
                }

                // Back
                Utility.polygon(
                    { x: Road.Back.x - wLeft * Road.Back.width,
                      y: Road.Back.y },
                    { x: Road.Back.x - wRight * Road.Back.width,
                      y: Road.Back.y },
                    { x: Road.Back.x - wRight * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - wLeft * Road.Back.width,
                      y: Road.Back.y - yBack },
                      Page.Colors.obstacle[1]);
                // Sides
                Utility.polygon(
                    { x: Road.Front.x - wLeft * Road.Front.width,
                      y: Road.Front.y },
                    { x: Road.Front.x - wLeft * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x - wLeft * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - wLeft * Road.Back.width,
                      y: Road.Back.y },
                      Page.Colors.obstacle[1]);
                Utility.polygon(
                    { x: Road.Front.x - wRight * Road.Front.width,
                      y: Road.Front.y },
                    { x: Road.Front.x - wRight * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x - wRight * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - wRight * Road.Back.width,
                      y: Road.Back.y },
                      Page.Colors.obstacle[1]);
                // Tops
                Utility.polygon(
                    { x: Road.Front.x - u - wLeft * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Front.x + u - wLeft * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x + v - wLeft * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - v - wLeft * Road.Back.width,
                      y: Road.Back.y - yBack},
                      Page.Colors.obstacle[0]);
                Utility.polygon(
                    { x: Road.Front.x - u - wRight * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Front.x + u - wRight * Road.Front.width,
                      y: Road.Front.y - yFront },
                    { x: Road.Back.x + v - wRight * Road.Back.width,
                      y: Road.Back.y - yBack },
                    { x: Road.Back.x - v - wRight * Road.Back.width,
                      y: Road.Back.y - yBack},
                      Page.Colors.obstacle[0]);

                if (!match) {
                    // Fronts
                    Utility.polygon(
                        { x: Road.Front.x - u - wLeft * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x + u - wRight * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x + u - wRight * Road.Front.width,
                          y: Road.Front.y - yFront },
                        { x: Road.Front.x - u - wLeft * Road.Front.width,
                          y: Road.Front.y - yFront },
                          Page.Colors.obstacle[0]);
                    Utility.polygon(
                        { x: Road.Front.x + u - wLeft * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x - u - wRight * Road.Front.width,
                          y: Road.Front.y },
                        { x: Road.Front.x - u - wRight * Road.Front.width,
                          y: Road.Front.y - yFront },
                        { x: Road.Front.x + u - wLeft * Road.Front.width,
                          y: Road.Front.y - yFront },
                          Page.Colors.obstacle[1]);
                }
            }
        }
    }

    // Roof
    private static roof(id: number, yFront: number, yBack: number): void {
        for (let i: number = 0; i < 5; i++) {
            let wLeft: number = 0;
            if (Road.Segments.array[id].roof[i] == '#') wLeft = 5 - i * 2;
            else continue;

            while (Road.Segments.array[id].roof[i] == '#') {
                i++;
            }
            i--;

            let wRight = 5 - (i + 1) * 2;

            let u: number = Road.Front.width * 0.03;
            let v: number = Road.Back.width * 0.03;

            // Roof
            Utility.polygon(
                { x: Road.Front.x - wLeft * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Front.x - wRight * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Back.x - wRight * Road.Back.width,
                  y: Road.Back.y - yBack },
                { x: Road.Back.x - wLeft * Road.Back.width,
                  y: Road.Back.y - yBack},
                  Page.Colors.obstacle[1]);
            // Stripes
            Utility.polygon(
                { x: Road.Front.x - u - wLeft * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Front.x + u - wLeft * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Back.x + v - wLeft * Road.Back.width,
                  y: Road.Back.y - yBack },
                { x: Road.Back.x - v - wLeft * Road.Back.width,
                  y: Road.Back.y - yBack},
                  Page.Colors.obstacle[0]);
            Utility.polygon(
                { x: Road.Front.x - u - wRight * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Front.x + u - wRight * Road.Front.width,
                  y: Road.Front.y - yFront },
                { x: Road.Back.x + v - wRight * Road.Back.width,
                  y: Road.Back.y - yBack },
                { x: Road.Back.x - v - wRight * Road.Back.width,
                  y: Road.Back.y - yBack},
                  Page.Colors.obstacle[0]);
        }
    }

    // Polygon
    private static polygon(wFront: number, wBack: number, wLeft: number, wRight: number,
                           color: string): void {
        Utility.polygon(
            { x: Road.Front.x - Road.Front.cos * (wFront - wLeft * Road.Front.width),
              y: Road.Front.y + Road.Front.sin * (wFront - wLeft * Road.Front.width) },
            { x: Road.Front.x + Road.Front.cos * (wFront + wRight * Road.Front.width),
              y: Road.Front.y - Road.Front.sin * (wFront + wRight * Road.Front.width) },
            { x: Road.Back.x + Road.Back.cos * (wBack + wRight * Road.Back.width),
              y: Road.Back.y - Road.Back.sin * (wBack + wRight * Road.Back.width) },
            { x: Road.Back.x - Road.Back.cos * (wBack - wLeft * Road.Back.width),
              y: Road.Back.y + Road.Back.sin * (wBack - wLeft * Road.Back.width) }, color);
    }

}

// ===================================================================================== Controllers

// -------------------------------------------------------------------------------------------- Page
class Page {

    // HTML canvas
    public static canvas: HTMLCanvasElement = document.getElementById('canvas') as HTMLCanvasElement;
    // HTML canvas context
    public static context: CanvasRenderingContext2D = Page.canvas.getContext('2d');
    // Image data
    public static image: HTMLImageElement = new Image();

    // Return the current time in milliseconds since the page loaded
    public static time(): number {
        return performance.now();
    }

    // Colors
    public static Colors = class {

        // Background
        public static background: string = '#42328B';
        // Road                         light                            dark
        public static road: string[] = ['#B42B3F', '#A62942', '#922445', '#812149'];
        // Obstacle
        public static obstacle: string[] = ['#FFFFFF', '#5441AD'];
        // Car                         white      dark + light purple   dark blue                        light blue
        public static car: string[] = ['#FFFFFF', '#42328B', '#5441AD', '#0062E1', '#0095EF', '#00AEFF', '#57D0FF'];

    }

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  Initialize, Update, Render

    // Render
    public static render(): void {
        Page.context.fillStyle = Page.Colors.background;
        Page.context.textAlign = 'center';
        Page.context.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    }

}

// ----------------------------------------------------------------------------------------- Utility
class Utility {

    // Render a four-sided polygon
    public static polygon(p1: Pixel, p2: Pixel, p3: Pixel, p4: Pixel, color: string): void {
        Page.context.fillStyle = color;
        Page.context.beginPath();
        Page.context.moveTo(p1.x, p1.y);
        Page.context.lineTo(p2.x, p2.y);
        Page.context.lineTo(p3.x, p3.y);
        Page.context.lineTo(p4.x, p4.y);
        Page.context.closePath();
        Page.context.fill();
    }

    // Quadratic ease in
    public static easeIn(a: number, b: number, p: number): number {
        return a + (b - a) * (p * p);
    }

    // Quadratic ease out
    public static easeOut(a: number, b: number, p: number): number {
        return a + (b - a) * (1 - (1 - p) * (1 - p));
    }

    // Quintic ease in
    public static easeInQuint(a: number, b: number, p: number): number {
        return a + (b - a) * (p * p * p * p);
    }

    // Quintic ease out
    public static easeOutQuint(a: number, b: number, p: number): number {
        return a + (b - a) * (1 - Math.pow(1 - p, 5));
    }

}

// --------------------------------------------------------------------------------------- Generator
class Generator {

    private static s: number[] = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 114, 114, 115, 115, 116,
        116, 117, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125,
        125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134,
        134, 135, 135, 136, 136, 137, 137 ];
    
    private static r: number[] = [ 138, 139, 140, 141, 142 ];

    // Templates for sections of road segments
    private static templates: Template[] = [
        // 0
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: '[....]' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },
        // 1
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: '[||||]' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },
        // 2
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: '[    ]' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },
        // 3
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: ' [..] ' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },
        // 4
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: ' [||] ' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },
        // 5
        { len:  64, col: 0, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i:  64, s: ' [  ] ' } ],
          obstacles: [ { i:  64, s: '      ' } ],
          roof:      [ { i:  64, s: '     ' } ],
          next: Generator.s },

        // 6
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 18, 30 ] },
        // 7
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.25, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 19, 31 ] },
        // 8
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 20, 32 ] },
        // 9
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 21, 33 ] },

        // 10
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 22, 34 ] },
        // 11
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.25, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 23, 35 ] },
        // 12
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 24, 36 ] },
        // 13
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 25, 37 ] },

        // 14
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 26, 38 ] },
        // 15
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.25, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 27, 39 ] },
        // 16
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 28, 40 ] },
        // 17
        { len: 32, col: 0, x: [ 0 ], y: [ 32, 32, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 29, 41 ] },

        // 18
        { len: 32, col: 0, x: [ 0 ], y: [ 0.1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 18, 30 ] },
        // 19
        { len: 32, col: 0, x: [ 0 ], y: [ 0.25 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 19, 31 ] },
        // 20
        { len: 32, col: 0, x: [ 0 ], y: [ 0.5 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 20, 32 ] },
        // 21
        { len: 32, col: 0, x: [ 0 ], y: [ 1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 21, 33 ] },

        // 22
        { len: 32, col: 0, x: [ 0 ], y: [ 0.1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 22, 34 ] },
        // 23
        { len: 32, col: 0, x: [ 0 ], y: [ 0.25 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 23, 35 ] },
        // 24
        { len: 32, col: 0, x: [ 0 ], y: [ 0.5 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 24, 36 ] },
        // 25
        { len: 32, col: 0, x: [ 0 ], y: [ 1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 25, 37 ] },

        // 26
        { len: 32, col: 0, x: [ 0 ], y: [ 0.1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 26, 38 ] },
        // 27
        { len: 32, col: 0, x: [ 0 ], y: [ 0.25 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 27, 39 ] },
        // 28
        { len: 32, col: 0, x: [ 0 ], y: [ 0.5 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 28, 40 ] },
        // 29
        { len: 32, col: 0, x: [ 0 ], y: [ 1 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 29, 41 ] },

        // 30
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 31
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 32
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 33
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },

        // 34
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 35
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 36
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 37
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },

        // 38
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 39
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 40
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },
        // 41
        { len: 32, col: 0, x: [ 0 ], y: [ 0, 0, 0, 1, 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 0, 1, 2, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ] },

        // 42
        { len: 32, col: 0, x: [ 32, 32, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 54, 66 ] },
        // 43
        { len: 32, col: 0, x: [ 32, 32, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 55, 67 ] },
        // 44
        { len: 32, col: 0, x: [ 32, 32, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 56, 68 ] },
        // 45
        { len: 32, col: 0, x: [ 32, 32, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 57, 69 ] },

        // 46
        { len: 32, col: 0, x: [ 32, 32, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 58, 70 ] },
        // 47
        { len: 32, col: 0, x: [ 32, 32, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 59, 71 ] },
        // 48
        { len: 32, col: 0, x: [ 32, 32, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 60, 72 ] },
        // 49
        { len: 32, col: 0, x: [ 32, 32, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 61, 73 ] },

        // 50
        { len: 32, col: 0, x: [ 32, 32, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 62, 74 ] },
        // 51
        { len: 32, col: 0, x: [ 32, 32, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 63, 75 ] },
        // 52
        { len: 32, col: 0, x: [ 32, 32, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 64, 76 ] },
        // 53
        { len: 32, col: 0, x: [ 32, 32, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 65, 77 ] },

        // 54
        { len: 32, col: 0, x: [ 0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 54, 66 ] },
        // 55
        { len: 32, col: 0, x: [ 0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 55, 67 ] },
        // 56
        { len: 32, col: 0, x: [ 0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 56, 68 ] },
        // 57
        { len: 32, col: 0, x: [ 0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 57, 69 ] },

        // 58
        { len: 32, col: 0, x: [ 0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 58, 70 ] },
        // 59
        { len: 32, col: 0, x: [ 0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 59, 71 ] },
        // 60
        { len: 32, col: 0, x: [ 0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 60, 72 ] },
        // 61
        { len: 32, col: 0, x: [ 0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 61, 73 ] },

        // 62
        { len: 32, col: 0, x: [ 0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 62, 74 ] },
        // 63
        { len: 32, col: 0, x: [ 0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 63, 75 ] },
        // 64
        { len: 32, col: 0, x: [ 0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 64, 76 ] },
        // 65
        { len: 32, col: 0, x: [ 0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 65, 77 ] },

        // 66
        { len: 32, col: 0, x: [ 0, 0, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 67
        { len: 32, col: 0, x: [ 0, 0, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 68
        { len: 32, col: 0, x: [ 0, 0, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 69
        { len: 32, col: 0, x: [ 0, 0, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 70
        { len: 32, col: 0, x: [ 0, 0, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 71
        { len: 32, col: 0, x: [ 0, 0, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 72
        { len: 32, col: 0, x: [ 0, 0, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 73
        { len: 32, col: 0, x: [ 0, 0, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 74
        { len: 32, col: 0, x: [ 0, 0, 0, 0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 75
        { len: 32, col: 0, x: [ 0, 0, 0, 0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 76
        { len: 32, col: 0, x: [ 0, 0, 0, 0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 77
        { len: 32, col: 0, x: [ 0, 0, 0, 0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 78
        { len: 32, col: 0, x: [ 32, 32, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 90, 102 ] },
        // 79
        { len: 32, col: 0, x: [ 32, 32, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 91, 103 ] },
        // 80
        { len: 32, col: 0, x: [ 32, 32, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 92, 104 ] },
        // 81
        { len: 32, col: 0, x: [ 32, 32, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 93, 105 ] },

        // 82
        { len: 32, col: 0, x: [ 32, 32, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 94, 106 ] },
        // 83
        { len: 32, col: 0, x: [ 32, 32, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 95, 107 ] },
        // 84
        { len: 32, col: 0, x: [ 32, 32, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 96, 108 ] },
        // 85
        { len: 32, col: 0, x: [ 32, 32, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 97, 109 ] },

        // 86
        { len: 32, col: 0, x: [ 32, 32, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 98, 110 ] },
        // 87
        { len: 32, col: 0, x: [ 32, 32, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 99, 111 ] },
        // 88
        { len: 32, col: 0, x: [ 32, 32, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 100, 112 ] },
        // 89
        { len: 32, col: 0, x: [ 32, 32, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 101, 113 ] },

        // 90
        { len: 32, col: 0, x: [ -0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 90, 102 ] },
        // 91
        { len: 32, col: 0, x: [ -0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 91, 103 ] },
        // 92
        { len: 32, col: 0, x: [ -0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 92, 104 ] },
        // 93
        { len: 32, col: 0, x: [ -0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 93, 105 ] },

        // 94
        { len: 32, col: 0, x: [ -0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 94, 106 ] },
        // 95
        { len: 32, col: 0, x: [ -0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 95, 107 ] },
        // 96
        { len: 32, col: 0, x: [ -0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 96, 108 ] },
        // 97
        { len: 32, col: 0, x: [ -0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 97, 109 ] },

        // 98
        { len: 32, col: 0, x: [ -0.03 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 98, 110 ] },
        // 99
        { len: 32, col: 0, x: [ -0.1 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 99, 111 ] },
        // 100
        { len: 32, col: 0, x: [ -0.25 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 100, 112 ] },
        // 101
        { len: 32, col: 0, x: [ -0.5 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: [ 101, 113 ] },

        // 102
        { len: 32, col: 0, x: [ 0, 0, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 103
        { len: 32, col: 0, x: [ 0, 0, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 104
        { len: 32, col: 0, x: [ 0, 0, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 105
        { len: 32, col: 0, x: [ 0, 0, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[....]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 106
        { len: 32, col: 0, x: [ 0, 0, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 107
        { len: 32, col: 0, x: [ 0, 0, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 108
        { len: 32, col: 0, x: [ 0, 0, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 109
        { len: 32, col: 0, x: [ 0, 0, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[||||]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 110
        { len: 32, col: 0, x: [ 0, 0, 0, -0.03, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 111
        { len: 32, col: 0, x: [ 0, 0, 0, -0.1, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 112
        { len: 32, col: 0, x: [ 0, 0, 0, -0.25, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },
        // 113
        { len: 32, col: 0, x: [ 0, 0, 0, -0.5, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 32, s: '[    ]' } ],
          obstacles: [ { i: 32, s: '      ' } ],
          roof:      [ { i: 32, s: '     ' } ],
          next: Generator.s },

        // 114
        { len: 256, col: 0, x: [ 0 ], y: [ 32, 220, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 1, s: ' .. , ' }, { i: 24, s: ' ,  , ' }, { i: 25, s: ' ,.., ' }, { i: 48, s: ' ,  , ' },
            { i: 49, s: ' ..oo ' }, { i: 72, s: ' ,  , ' }, { i: 73, s: ' ..., ' }, { i: 96, s: ' ,  , ' },
            { i: 97, s: ' oo.. ' }, { i: 116, s: ' ,  , ' }, { i: 117, s: ' ..., ' }, { i: 152, s: ' ,  , ' },
            { i: 153, s: ' ,ooo ' }, { i: 176, s: ' ,  , ' }, { i: 177, s: ' ..oo ' }, { i: 200, s: ' ,  , ' },
            { i: 201, s: ' ,ooo ' }, { i: 224, s: ' ,  , ' }, { i: 225, s: ' .. , ' }, { i: 255, s: ' ,  , ' },
            { i: 256, s: ' oo , ' },
            ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 115
        { len: 256, col: 0, x: [ 0 ], y: [ 32, 220, 0, -0.05, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 1, s: ' .. , ' }, { i: 24, s: ' ,  , ' }, { i: 25, s: ' ,.., ' }, { i: 48, s: ' ,  , ' },
            { i: 49, s: ' ..oo ' }, { i: 72, s: ' ,  , ' }, { i: 73, s: ' ..., ' }, { i: 96, s: ' ,  , ' },
            { i: 97, s: ' oo.. ' }, { i: 116, s: ' ,  , ' }, { i: 117, s: ' ..., ' }, { i: 152, s: ' ,  , ' },
            { i: 153, s: ' ,ooo ' }, { i: 176, s: ' ,  , ' }, { i: 177, s: ' ..oo ' }, { i: 200, s: ' ,  , ' },
            { i: 201, s: ' ,ooo ' }, { i: 224, s: ' ,  , ' }, { i: 225, s: ' .. , ' }, { i: 255, s: ' ,  , ' },
            { i: 256, s: ' oo , ' },
            ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 116
        { len: 256, col: 0, x: [ 0 ], y: [ 32, 220, 0, -0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 1, s: ' .. , ' }, { i: 24, s: ' ,  , ' }, { i: 25, s: ' ,.., ' }, { i: 48, s: ' ,  , ' },
            { i: 49, s: ' ..oo ' }, { i: 72, s: ' ,  , ' }, { i: 73, s: ' ..., ' }, { i: 96, s: ' ,  , ' },
            { i: 97, s: ' oo.. ' }, { i: 116, s: ' ,  , ' }, { i: 117, s: ' ..., ' }, { i: 152, s: ' ,  , ' },
            { i: 153, s: ' ,ooo ' }, { i: 176, s: ' ,  , ' }, { i: 177, s: ' ..oo ' }, { i: 200, s: ' ,  , ' },
            { i: 201, s: ' ,ooo ' }, { i: 224, s: ' ,  , ' }, { i: 225, s: ' .. , ' }, { i: 255, s: ' ,  , ' },
            { i: 256, s: ' oo , ' },
            ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 117
        { len: 256, col: 0, x: [ 0 ], y: [ 64, 180, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 1, s: ' , .. ' }, { i: 24, s: ' ,  , ' }, { i: 25, s: ' ,.., ' }, { i: 48, s: ' ,  , ' },
            { i: 49, s: ' OOoo ' }, { i: 72, s: ' ,  , ' }, { i: 73, s: ' ,ooo ' }, { i: 96, s: ' ,  , ' },
            { i: 97, s: ' ..OO ' }, { i: 116, s: ' ,  , ' }, { i: 117, s: ' ..., ' }, { i: 152, s: ' ,  , ' },
            { i: 153, s: ' ..., ' }, { i: 176, s: ' ,  , ' }, { i: 177, s: ' OO.. ' }, { i: 200, s: ' ,  , ' },
            { i: 201, s: ' ..., ' }, { i: 224, s: ' ,  , ' }, { i: 225, s: ' oo , ' }, { i: 255, s: ' ,  , ' },
            { i: 256, s: ' oo , ' },
            ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 118
        { len: 256, col: 0, x: [ 64, 196, 0, -0.05, 0 ], y: [ 32, 220, 0, 0.3, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 1, s: '..  ..' }, { i: 24, s: ' ,  , ' }, { i: 25, s: ' ,.., ' }, { i: 48, s: ' ,  , ' },
            { i: 49, s: ' , oo ' }, { i: 72, s: ' ,  , ' }, { i: 73, s: ' ..oo ' }, { i: 96, s: ' ,  , ' },
            { i: 97, s: ' ooo, ' }, { i: 120, s: ' ,  , ' }, { i: 121, s: ' ,oo, ' }, { i: 144, s: ' ,  , ' },
            { i: 145, s: ' , oo ' }, { i: 168, s: ' ,  , ' }, { i: 169, s: ' ,... ' }, { i: 192, s: ' ,  , ' },
            { i: 193, s: ' oo , ' }, { i: 216, s: ' ,  , ' }, { i: 217, s: ' ooo, ' }, { i: 234, s: ' ,  , ' },
            { i: 235, s: ' ,.., ' }, { i: 254, s: ' ,  , ' }, { i: 255, s: ' oo , ' }
            ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 119
        { len: 256, col: 0, x: [ 0 ], y: [ 32, 200, 0, 0.4, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 16, s: ' , oo ' }, { i: 24, s: ' ,ooo ' }, { i: 40, s: ' , oo ' }, { i: 56, s: ' oo , ' },
            { i: 64, s: ' ooo, ' }, { i: 80, s: ' oo , ' }, { i: 96, s: ' , oo ' }, { i: 104, s: ' ,ooo ' },
            { i: 120, s: ' , oo ' }, { i: 136, s: ' oo , ' }, { i: 144, s: ' ooo, ' }, { i: 160, s: ' oo , ' },
            { i: 176, s: ' , oo ' }, { i: 184, s: ' ,ooo ' }, { i: 200, s: ' , oo ' }, { i: 216, s: ' oo , ' },
            { i: 224, s: ' ooo, ' }, { i: 240, s: ' oo , ' }, { i: 256, s: ' ,  , '}
        ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 120
        { len: 256, col: 0, x: [ 64, 180, 0, 0.05, 0 ], y: [ 32, 200, 0, 0.2, 0 ], r: [ 0 ],
          lanes:     [ { i: 256, s: ' [||] ' } ],
          obstacles: [
            { i: 16, s: ' .. , ' }, { i: 24, s: ' ..., ' }, { i: 40, s: ' .. , ' }, { i: 56, s: ' , .. ' },
            { i: 64, s: ' ,... ' }, { i: 80, s: ' , .. ' }, { i: 96, s: ' .. , ' }, { i: 104, s: ' ..., ' },
            { i: 120, s: ' .. , ' }, { i: 136, s: ' , .. ' }, { i: 144, s: ' ,... ' }, { i: 160, s: ' , .. ' },
            { i: 176, s: ' .. , ' }, { i: 184, s: ' ..., ' }, { i: 200, s: ' .. , ' }, { i: 216, s: ' , .. ' },
            { i: 224, s: ' ,... ' }, { i: 240, s: ' , .. ' }, { i: 256, s: ' .. , '}
        ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },

        // 121
        { len: 256, col: 1, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 288, s: '[ || ]' } ],
          obstacles: [ { i: 1, s: 'Q OO Q'},
            { i: 31, s: 'Q    Q' }, { i: 32, s: 'OOO  Q'}, { i: 63, s: 'Q .. Q' }, { i: 64, s: 'Q OO Q'},
            { i: 95, s: 'Q    Q' }, { i: 96, s: 'Q  OOO'}, { i: 127, s: 'Q .. Q' }, { i: 128, s: 'Q OO Q'},
            { i: 159, s: 'Q .. Q' }, { i: 160, s: 'QOOOOQ'}, { i: 191, s: 'Q .. Q' }, { i: 192, s: 'Q OO Q'},
            { i: 223, s: 'Q    Q' }, { i: 224, s: 'OOO  Q'}, { i: 256, s: 'Q    Q'}
        ],
          roof:      [ { i: 288, s: '#####' } ],
          next: Generator.s },
        // 122
        { len: 256, col: 1, x: [ 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 288, s: '[ || ]' } ],
          obstacles: [ { i: 1, s: 'Q OOOO'},
            { i: 31, s: 'Q    Q' }, { i: 32, s: 'OOOO Q'}, { i: 63, s: 'Q    Q' }, { i: 64, s: 'Q  OOO'},
            { i: 95, s: 'Q    Q' }, { i: 96, s: 'OOO OO'}, { i: 127, s: 'Q .. Q' }, { i: 128, s: 'Q OO Q'},
            { i: 159, s: 'Q    Q' }, { i: 160, s: 'Q  OOO'}, { i: 191, s: 'Q .. Q' }, { i: 192, s: 'Q OO Q'},
            { i: 223, s: 'Q  , Q' }, { i: 224, s: 'OOO, Q'}, { i: 256, s: 'Q  , Q'}
        ],
          roof:      [ { i: 288, s: '#####' } ],
          next: Generator.s },
        // 123
        { len: 256, col: 1, x: [ 32, 224, 0, 0.15, 0 ], y: [ 0 ], r: [ 0 ],
          lanes:     [ { i: 288, s: '[ || ]' } ],
          obstacles: [ { i: 1, s: 'Q OO Q'},
            { i: 31, s: 'Q    Q' }, { i: 32, s: 'OOO  Q'}, { i: 63, s: 'Q .. Q' }, { i: 64, s: 'Q OO Q'},
            { i: 95, s: 'Q    Q' }, { i: 96, s: 'Q  OOO'}, { i: 127, s: 'Q .. Q' }, { i: 128, s: 'Q OO Q'},
            { i: 159, s: 'Q    Q' }, { i: 160, s: 'QOO  Q'}, { i: 191, s: 'Q .. Q' }, { i: 192, s: 'Q OO Q'},
            { i: 223, s: 'Q    Q' }, { i: 224, s: 'O  OOQ'}, { i: 256, s: 'Q    Q'}
        ],
          roof:      [ { i: 288, s: '#####' } ],
          next: Generator.s },
        // 124
        { len: 256, col: 1, x: [ 32, 224, 0, -0.15, 0 ], y: [ 64, 192, 0, 0.1, 0 ], r: [ 0 ],
          lanes:     [ { i: 288, s: '[ || ]' } ],
          obstacles: [ { i: 1, s: 'Q OO Q'},
            { i: 31, s: 'Q    Q' }, { i: 32, s: 'QOOOOQ'}, { i: 63, s: 'Q .. Q' }, { i: 64, s: 'Q OO Q'},
            { i: 95, s: 'Q    Q' }, { i: 96, s: 'OO..OO'}, { i: 127, s: 'Q .. Q' }, { i: 128, s: 'Q OO Q'},
            { i: 159, s: 'Q    Q' }, { i: 160, s: 'Q  OOO'}, { i: 191, s: 'Q .. Q' }, { i: 192, s: 'Q OO Q'},
            { i: 223, s: 'Q    Q' }, { i: 224, s: 'OOO  Q'}, { i: 256, s: 'Q    Q'}
        ],
          roof:      [ { i: 288, s: '#####' } ],
          next: Generator.s },
        // 125
        { len: 256, col: 1, x: [ 0 ], y: [ 32, 224, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [ { i: 288, s: '[ || ]' } ],
          obstacles: [ { i: 1, s: 'Q OOOO'},
            { i: 31, s: 'Q    Q' }, { i: 32, s: 'OOOO Q'}, { i: 63, s: 'Q    Q' }, { i: 64, s: 'Q  OOO'},
            { i: 95, s: 'Q    Q' }, { i: 96, s: 'OOO OO'}, { i: 127, s: 'Q .. Q' }, { i: 128, s: 'Q OO Q'},
            { i: 159, s: 'Q    Q' }, { i: 160, s: 'Q  OOO'}, { i: 191, s: 'Q .. Q' }, { i: 192, s: 'Q OO Q'},
            { i: 223, s: 'Q  ..Q' }, { i: 224, s: 'OOO..Q'}, { i: 256, s: 'Q  ..Q'}
        ],
          roof:      [ { i: 288, s: '#####' } ],
          next: Generator.s },

        // 126
        { len: 64, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 32, 0, Math.PI / 128, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 127
        { len: 64, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 32, 0, -Math.PI / 128, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 128
        { len: 96, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 64, 0, Math.PI / 128, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },
        // 129
        { len: 96, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 64, 0, -Math.PI / 128, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },
        // 130
        { len: 64, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 32, 0, Math.PI / 64, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 131
        { len: 64, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 32, 0, -Math.PI / 64, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 132
        { len: 96, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 64, 0, Math.PI / 64, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },
        // 133
        { len: 96, col: 1, x: [ 0 ], y: [ 0 ], r: [ 32, 64, 0, -Math.PI / 64, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },
        // 134
        { len: 64, col: 1, x: [ 32, 32, 0, -0.2, 0 ], y: [ 32, 32, 0, 0.2, 0 ], r: [ 32, 32, 0, Math.PI / 128, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 135
        { len: 64, col: 1, x: [ 32, 32, 0, 0.2, 0 ], y: [ 32, 32, 0, 0.2, 0 ], r: [ 32, 32, 0, -Math.PI / 128, 0 ],
          lanes:     [ { i: 64, s: '[||||]' } ],
          obstacles: [ { i: 64, s: '      ' } ],
          roof:      [ { i: 64, s: '     ' } ],
          next: Generator.r },
        // 136
        { len: 96, col: 1, x: [ 32, 64, 0, -0.2, 0 ], y: [ 32, 64, 0, 0.2, 0 ], r: [ 32, 64, 0, Math.PI / 128, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },
        // 137
        { len: 96, col: 1, x: [ 32, 64, 0, 0.2, 0 ], y: [ 32, 64, 0, 0.2, 0 ], r: [ 32, 64, 0, -Math.PI / 128, 0 ],
          lanes:     [ { i: 96, s: '[||||]' } ],
          obstacles: [ { i: 96, s: '      ' } ],
          roof:      [ { i: 96, s: '     ' } ],
          next: Generator.r },

        // 138
        { len: 128, col: 1, x: [ 32, 96, 0, 0.3, 0 ], y: [ 32, 96, 0, 0.3, 0 ], r: [ 0 ],
          lanes:     [
            { i: 16, s: '[|][|]' }, { i: 32, s: '[]  []' }, { i: 48, s: '[|][|]' }, { i: 64, s: ' [||] ' },
            { i: 80, s: '  []  ' }, { i: 96, s: ' [||] ' }, { i: 112, s: '[|][|]' }, { i: 128, s: '[||||]' },
            ],
          obstacles: [ { i: 128, s: '      ' } ],
          roof:      [ { i: 128, s: '     ' } ],
          next: Generator.s },
        // 139
        { len: 128, col: 1, x: [ 32, 96, 0, -0.4, 0 ], y: [ 32, 96, 0, 0.3, 0 ], r: [ 0 ],
          lanes:     [
            { i: 16, s: '[|][|]' }, { i: 32, s: '[]  []' }, { i: 48, s: '[|][|]' }, { i: 64, s: ' [||] ' },
            { i: 80, s: '  []  ' }, { i: 96, s: ' [||] ' }, { i: 112, s: '[|][|]' }, { i: 128, s: '[||||]' },
            ],
          obstacles: [ { i: 128, s: '      ' } ],
          roof:      [ { i: 128, s: '     ' } ],
          next: Generator.s },
        // 140
        { len: 256, col: 1, x: [ 32, 224, 0, 0.2, 0 ], y: [ 32, 224, 0, 0.4, 0 ], r: [ 0 ],
          lanes:     [
            { i: 32, s: '   [|]' }, { i: 64, s: '[||||]' }, { i: 96, s: '[|]   ' }, { i: 128, s: '[||||]' },
            { i: 160, s: '   [|]' }, { i: 192, s: '[||||]' }, { i: 224, s: '[|]   ' }, { i: 256, s: '[||||]' },
            ],
          obstacles: [ { i: 256, s: '      ' } ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 141
        { len: 256, col: 1, x: [ 32, 224, 0, -0.35, 0 ], y: [ 32, 224, 0, 0.4, 0 ], r: [ 0 ],
          lanes:     [
            { i: 32, s: '   [|]' }, { i: 64, s: '[||||]' }, { i: 96, s: '[|]   ' }, { i: 128, s: '[||||]' },
            { i: 160, s: '   [|]' }, { i: 192, s: '[||||]' }, { i: 224, s: '[|]   ' }, { i: 256, s: '[||||]' },
            ],
          obstacles: [ { i: 256, s: '      ' } ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
        // 142
        { len: 256, col: 1, x: [ 64, 192, 0, 0.1, 0 ], y: [ 64, 192, 0, 0.5, 0 ], r: [ 0 ],
          lanes:     [
            { i: 32, s: ' [||] ' }, { i: 64, s: '[|][|]' }, { i: 96, s: ' [||] ' }, { i: 112, s: '[|][|]' },
            { i: 148, s: '   [|]' }, { i: 192, s: '[|||] ' }, { i: 224, s: '[|]   ' }, { i: 256, s: '[||||]' },
            ],
          obstacles: [ { i: 256, s: '      ' } ],
          roof:      [ { i: 256, s: '     ' } ],
          next: Generator.s },
    ];

    // Id of the current template
    //public static template: number = 142;
    public static template: number = Generator.s[Math.random() * Generator.s.length >> 0];
    // Id of the current segment
    public static segment: number = 0;
    // Id of the current lane block   // Id of the current obstacle block
    public static lane: number = 0;   public static obstacle: number = 0;
    // Id of the current roof block   // Id of the current color block
    public static roof: number = 0;   public static color: number = 0;

    public static generate(id: number): void {
        let s: Segment = Road.Segments.array[(id + Road.Segments.count - 3) % Road.Segments.count];
        let t: Template = Generator.templates[Generator.template];

        // Color
        if (t.col == 0) {
            s.color = Generator.segment % 16 < 8 ? 0 : 1;
        } else {
            s.color = Generator.segment % 16 < 8 ? 3 : 2;
        }

        // Delta of X
        if (t.x.length == 1) s.deltaX = t.x[0];
        else {
            if (Generator.segment < t.x[0]) s.deltaX = Utility.easeIn(t.x[2], t.x[3], Generator.segment / t.x[0]);
            else if (Generator.segment < t.x[1]) s.deltaX = t.x[3];
            else s.deltaX = Utility.easeOut(t.x[3], t.x[4], (Generator.segment - t.x[1]) / (t.len - t.x[1]));
        }

        // Delta of Y
        if (t.y.length == 1) s.deltaY = t.y[0];
        else {
            if (Generator.segment < t.y[0]) s.deltaY = Utility.easeIn(t.y[2], t.y[3], Generator.segment / t.y[0]);
            else if (Generator.segment < t.y[1]) s.deltaY = t.y[3];
            else s.deltaY = Utility.easeOut(t.y[3], t.y[4], (Generator.segment - t.y[1]) / (t.len - t.y[1]));
        }

        // Delta of R
        if (t.r.length == 1) s.deltaR = t.r[0];
        else {
            if (Generator.segment < t.r[0]) s.deltaR = Utility.easeIn(t.r[2], t.r[3], Generator.segment / t.r[0]);
            else if (Generator.segment < t.r[1]) s.deltaR = t.r[3];
            else s.deltaR = Utility.easeOut(t.r[3], t.r[4], (Generator.segment - t.r[1]) / (t.len - t.r[1]));
        }

        // Lanes
        if (Generator.segment >= t.lanes[Generator.lane].i) Generator.lane++;
        s.lanes = t.lanes[Generator.lane].s;

        // Obstacles
        if (Generator.segment >= t.obstacles[Generator.obstacle].i) Generator.obstacle++;
        s.obstacles = t.obstacles[Generator.obstacle].s;

        // Roof
        if (Generator.segment >= t.roof[Generator.roof].i) Generator.roof++;
        s.roof = t.roof[Generator.roof].s;

        // Choose next template
        Generator.segment++;
        if (Generator.segment + 1 == t.len) {
            Generator.segment = 0;
            Generator.template = t.next[Math.random() * t.next.length >> 0];
            Generator.lane = 0;
            Generator.obstacle = 0;
            Generator.roof = 0;
        }
    }

}

// -------------------------------------------------------------------------------------------- Game
class Game {

    // Most recent time callback             // Previous time callback
    private static timeNow: number = null;   private static timeLast: number = Page.time();
    // Time per frame in seconds                  // Elapsed time since last callback
    private static timeFrame: number = 1 / FPS;   private static timeDelta: number = null;
    // Accumulated time since last frame, used for fixed time step in update
    private static timeAccumulator: number = null;
    // Whether the game loop is running       // Whether the game controller is paused
    public static running: boolean = false;   public static paused: boolean = true;
    // Counter used for cinematics
    public static counter: number = 0;

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  Initialize, Update, Render

    // Initialize
    public static initialize(): void {
        Player.initialize();
        Road.initialize();
        Page.render();
        Page.context.font = 'bold 48px sans-serif';
        let w: number = Page.context.measureText('Klikni a hrej!').width;
        Utility.polygon({ x: FRAME_WH - w / 2 - 15, y: 10 }, { x: FRAME_WH + w / 2 + 15, y: 10 },
                        { x: FRAME_WH + w / 2 + 15, y: 69 }, { x: FRAME_WH - w / 2 - 15, y: 69 },
                        Page.Colors.obstacle[1]);
        Page.context.fillStyle = 'white';
        Page.context.fillText('Klikni a hrej!', FRAME_WH, 58);

        Utility.polygon({ x: FRAME_WH + w / 2 + 25, y: 10 }, { x: FRAME_WH + w / 2 + 55 + 83, y: 10 },
                        { x: FRAME_WH + w / 2 + 55 + 83, y: 69 }, { x: FRAME_WH + w / 2 + 25, y: 69 },
                        Page.Colors.obstacle[1]);
        Page.context.drawImage(Page.image, 1, 1, 83, 36, FRAME_WH + w / 2 + 40, 22, 83, 36);

        Utility.polygon({ x: FRAME_WH - w / 2 - 25, y: 10 }, { x: FRAME_WH - w / 2 - 45 - 131, y: 10 },
                        { x: FRAME_WH - w / 2 - 45 - 131, y: 69 }, { x: FRAME_WH - w / 2 - 25, y: 69 },
                        Page.Colors.obstacle[1]);
        Page.context.drawImage(Page.image, 86, 1, 131, 40, FRAME_WH - w / 2 - 35 - 131, 22, 131, 40);

        Page.context.drawImage(Page.image, 0, 42, 766, 353, FRAME_WH - 383, FRAME_HH - 176, 766, 353);
    }

    // Start the game loop
    public static start(): void {
        Game.counter = 0;
        Game.running = true;
        Game.loop();
    }

    // Game restart
    public static restart(): void {
        Game.counter = 0;
        Player.distance = 1;
        Player.z = 0;
        Player.x = 0;
        Player.baseZ = 38;
        Game.paused = true;
        Game.running = false;
        Generator.template = 0;
        Generator.segment = 0;
        Generator.lane = 0;
        Generator.obstacle = 0;
        Generator.roof = 0;
        Generator.color = 0;
        Game.initialize();
    }

    // Game loop
    public static loop(): void {
        Game.timeNow = Page.time();
        // Restrict maximum delta time to 1 second, in case of large delay due to tab switch, etc.
        Game.timeDelta = Math.min((Game.timeNow - Game.timeLast) / 1000, 1);
        Game.timeLast = Game.timeNow;
        Game.timeAccumulator += Game.timeDelta;
        // Read https://www.gafferongames.com/post/fix_your_timestep/
        while (Game.timeAccumulator >= Game.timeFrame) {
            Game.update();
            Game.timeAccumulator -= Game.timeFrame;
        }
        Game.render();

        // Schedule game loop call just before the browser repaints
        if (Game.counter != 799)
            window.requestAnimationFrame(Game.loop.bind(Game));
    }

    // Update game state
    private static update(): void {
        if (Game.paused) {
            //Game.counter = 420; // BYPASS starting cinematic by uncommenting this line

            // Camera flydown
            if (Game.counter < 120) {
                Player.Camera.height = Utility.easeOutQuint(1000, 50, Game.counter / 150);
            // Camera height reset
            } else if (Game.counter == 120) {
                Player.Camera.height = 50;
            // Car accelerating
            } else if (Game.counter > 120 && Game.counter < 170) {
                Player.speed += 0.4;
                Player.update();
            } else if (Game.counter == 170) {
                Game.counter = 419;
            // Controls enabled, game unpaused
            } else if (Game.counter == 420) {
                Player.distance = Math.round(Player.distance);
                Player.speed = 20;
                Game.paused = false;
            // Crash happened, smoothly move out the car from the scene and return
            } else if (Game.counter < 620) {
                if (Game.counter == 422) Player.baseZ = 38;
                else if (Game.counter == 423) Player.baseZ = 31;
                else if (Game.counter == 424) Player.baseZ = 24;
                else if (Game.counter == 425) Player.baseZ = 17;
                else if (Game.counter == 426) Player.baseZ = 10;
                else if (Game.counter == 427) Player.baseZ = 5;
                else if (Game.counter == 428) Player.baseZ = 0;
                Player.speed -= 0.1;
                Player.update();
            } else if (Game.counter == 620) {
                Player.speed = 0;
                Player.update();
            }
            Game.counter++;
        } else {
            Player.update();
            Game.collision();
        }
    }

    // Render
    public static render(): void {
        Page.render();
        Road.render();

        if (Game.paused) {
            if (Game.counter < 50) {
                Page.context.globalAlpha = Utility.easeInQuint(1, 0, Game.counter / 50);
                Page.render();
                Page.context.globalAlpha = 1;
            } else if (Game.counter > 700 && Game.counter < 800) {
                Page.context.globalAlpha = Utility.easeOutQuint(0, 1, (Game.counter - 700) / 100);
                Page.render();
                Page.context.globalAlpha = 1;
            }
        }

        Game.score();
    }

    //  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   Private

    // Score
    private static score(): void {
        Page.context.font = 'bold 48px sans-serif';

        let w: number = Page.context.measureText(Player.distance.toString()).width;
        if (Game.counter >= 422) w = Page.context.measureText('Kolize').width;

        Utility.polygon({ x: FRAME_WH - w / 2 - 45 - 136, y: 5 }, { x: FRAME_WH + w / 2 + 55 + 88, y: 5 },
                        { x: FRAME_WH + w / 2 + 55 + 88, y: 74 }, { x: FRAME_WH - w / 2 - 45 - 136, y: 74 },
                        Page.Colors.background);

        Utility.polygon({ x: FRAME_WH - w / 2 - 15, y: 10 }, { x: FRAME_WH + w / 2 + 15, y: 10 },
                        { x: FRAME_WH + w / 2 + 15, y: 69 }, { x: FRAME_WH - w / 2 - 15, y: 69 },
                        Page.Colors.obstacle[1]);
        if (Game.counter < 422) {
            Page.context.fillStyle = 'white';
            Page.context.fillText((Player.distance - 1).toString(), FRAME_WH, 58);
        } else {
            Page.context.fillStyle = Page.Colors.road[0];
            Page.context.fillText('Kolize', FRAME_WH, 58);
        }

        Utility.polygon({ x: FRAME_WH + w / 2 + 25, y: 10 }, { x: FRAME_WH + w / 2 + 55 + 83, y: 10 },
                        { x: FRAME_WH + w / 2 + 55 + 83, y: 69 }, { x: FRAME_WH + w / 2 + 25, y: 69 },
                        Page.Colors.obstacle[1]);
        Page.context.drawImage(Page.image, 1, 1, 83, 36, FRAME_WH + w / 2 + 40, 22, 83, 36);

        Utility.polygon({ x: FRAME_WH - w / 2 - 25, y: 10 }, { x: FRAME_WH - w / 2 - 45 - 131, y: 10 },
                        { x: FRAME_WH - w / 2 - 45 - 131, y: 69 }, { x: FRAME_WH - w / 2 - 25, y: 69 },
                        Page.Colors.obstacle[1]);
        Page.context.drawImage(Page.image, 86, 1, 131, 40, FRAME_WH - w / 2 - 35 - 131, 22, 131, 40);

        if (Game.counter > 620) {
            Page.context.font = 'bold 128px sans-serif';
            w = Page.context.measureText((Player.distance - 1).toString()).width;
            Utility.polygon({ x: FRAME_WH - w / 2 - 30, y: FRAME_HH - 125 }, { x: FRAME_WH + w / 2 + 30, y: FRAME_HH - 125 },
                            { x: FRAME_WH + w / 2 + 30, y: FRAME_HH + 35 }, { x: FRAME_WH - w / 2 - 30, y: FRAME_HH + 35 },
                            Page.Colors.background);
            Utility.polygon({ x: FRAME_WH - w / 2 - 25, y: FRAME_HH - 120 }, { x: FRAME_WH + w / 2 + 25, y: FRAME_HH - 120 },
                            { x: FRAME_WH + w / 2 + 25, y: FRAME_HH + 30 }, { x: FRAME_WH - w / 2 - 25, y: FRAME_HH + 30 },
                            Page.Colors.obstacle[1]);

            Page.context.fillStyle = 'white';
            Page.context.fillText((Player.distance - 1).toString(), FRAME_WH, FRAME_HH);

            Page.context.globalAlpha = 1;
            if (Game.counter == 799)
                Page.context.drawImage(Page.image, 0, 396, 534, 154, FRAME_WH - 187, FRAME_HH + 145, 374, 108);
        }
    }

    // Check for collisions
    private static collision(): void {
        // Check if the car is in a hole
        let inHole: boolean = true;
        for (let i: number = 0; i < 5; i++) {
            if (Road.Segments.array[Player.distance % Road.Segments.count].lanes[i] == '[') {
                let u: number = Road.Holes[i];
                while (Road.Segments.array[Player.distance % Road.Segments.count].lanes[i] != ']') {
                    i++;
                }
                let v: number = Road.Holes[i + 4];

                if (Player.x > u && Player.x < v)
                    inHole = false;
            }
        }

        // Check if the car is in an obstacle
        let inObstacle: boolean = false;
        for (let i: number = 0; i < 6; i++) {
            let obstacle: string = Road.Segments.array[Player.distance % Road.Segments.count].obstacles[i];
            if (obstacle == ',' || obstacle == 'q' || obstacle == 'Q') {
                if (Player.x > -Road.Obstacles[5 - i] && Player.x < Road.Obstacles[i])
                    inObstacle = true;
            }

            if (obstacle == '.' || obstacle == 'o' || obstacle == 'O') {
                let u: number = -Road.Obstacles[5 - i];
                while (Road.Segments.array[Player.distance % Road.Segments.count].obstacles[i] == obstacle) {
                    i++;
                }
                i--;
                let v: number = Road.Obstacles[i];

                if (Player.x > u && Player.x < v)
                    inObstacle = true;
            }
        }

        if (inHole || inObstacle) {
            Game.paused = true;
            Game.counter++;
        }
    }

}

// ============================================================= Start the engine, gas to the floor!

// If the browser does not support requestAnimationFrame, use setTimeout as fallback
if (!window.requestAnimationFrame) {
    window.requestAnimationFrame = function(callback: FrameRequestCallback): number {
        return window.setTimeout(callback, 1000 / FPS);
    }
}

// Resize the canvas to fit the screen
window.addEventListener('resize', function(event: UIEvent): void {
    Page.canvas.width = window.innerWidth < FRAME_WIDTH ? window.innerWidth - 20 : FRAME_WIDTH;
    Page.canvas.height = Page.canvas.width * FRAME_HEIGHT / FRAME_WIDTH;
    if (Page.canvas.height > window.innerHeight - 20) {
        Page.canvas.height = window.innerHeight - 20;
        Page.canvas.width = Page.canvas.height * FRAME_WIDTH / FRAME_HEIGHT;
    }
    Page.context.scale(Page.canvas.width / FRAME_WIDTH, Page.canvas.height / FRAME_HEIGHT);
    Game.render();
});
window.dispatchEvent(new Event('resize'));

// Load image and start the game when it is loaded
Page.image.onload = function() { Game.initialize(); };
Page.image.src = 'img.png';
