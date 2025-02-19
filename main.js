// main.js

// Set canvas dimensions.
const [width, height] = [window.innerWidth, window.innerHeight];

// Global click effect variables.
let clickEffect = 0.0;
let clickPos = vec3.create();

// Global mouse state.
let mouseIsDown = false;
let lastMouseX = 0;
let lastMouseY = 0;

// We'll update these every frame so the click can be unprojected properly.
let currentProjection = mat4.create();
let currentView = mat4.create();

// -------------------------
// Helper Functions
// -------------------------
function hslToRgb(h, s, l) {
  h /= 360;
  let r, g, b;
  if (!s) { 
    r = g = b = l; 
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [r, g, b];
}
const mix = (a, b, t) => a * (1 - t) + b * t;
const random = (min, max) => Math.random() * (max - min) + min;
function resizeCanvas(canvas, gl, w, h) {
  canvas.width = w;
  canvas.height = h;
  gl.viewport(0, 0, w, h);
}

// -------------------------
// Shader Helper Functions
// -------------------------
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}
function createProgram(gl, vsSource, fsSource) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

// -------------------------
// Texture Creation Functions
// -------------------------
function createStarTexture(gl) {
  const size = 64;
  const starCanvas = document.createElement("canvas");
  starCanvas.width = starCanvas.height = size;
  const ctx = starCanvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.5, "rgba(255,255,255,0.5)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                gl.UNSIGNED_BYTE, starCanvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}
function createTempTexture(gl, width, height) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

// -------------------------
// Unproject Function for Clicks
// -------------------------
function getWorldCoord(x, y, projection, view) {
  // Convert click (x,y) to normalized device coordinates.
  let ndcX = (x / width) * 2 - 1;
  let ndcY = 1 - (y / height) * 2;
  // Compute inverse of projection*view.
  let invPV = mat4.create();
  let pv = mat4.create();
  mat4.multiply(pv, projection, view);
  mat4.invert(invPV, pv);
  // Define points on the near and far plane.
  let nearPoint = vec3.fromValues(ndcX, ndcY, -1);
  let farPoint = vec3.fromValues(ndcX, ndcY, 1);
  let nearWorld = vec3.create();
  let farWorld = vec3.create();
  vec3.transformMat4(nearWorld, nearPoint, invPV);
  vec3.transformMat4(farWorld, farPoint, invPV);
  let rayDir = vec3.create();
  vec3.subtract(rayDir, farWorld, nearWorld);
  vec3.normalize(rayDir, rayDir);
  // Intersect with the plane z = 0.
  let t = -nearWorld[2] / rayDir[2];
  let worldPos = vec3.create();
  vec3.scaleAndAdd(worldPos, nearWorld, rayDir, t);
  return worldPos;
}

// -------------------------
// Shader Sources
// -------------------------

// Simulation Vertex Shader – note the new uniforms for click effect.
const simVert = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec3 aPrevPosition;
in vec4 aColor;
uniform float uTime, uRotationSpeed, uPrimaryRotationSpeed, uSecondaryRotationSpeed, uBlackHoleSize;
uniform mat4 uProjectionMatrix, uViewMatrix;
// Uniforms for click/swarm effect.
uniform vec3 uClickPos;
uniform float uClickEffect;
out vec4 vColor;
out vec3 vPrevPos, vCurrentPos;
 
// Simplex noise functions
vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g, l.zxy);
  vec3 i2 = max(g, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
         + i.y + vec4(0.0, i1.y, i2.y, 1.0))
         + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m*m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
void main(){
  vec3 pos = aPosition;
  bool isBlackHole = (pos.x == 0.0 && pos.y == 0.0 && pos.z == 0.0);
  float dist = length(pos);
  if(!isBlackHole){
    // Primary rotation.
    float primary = uTime * uPrimaryRotationSpeed;
    mat3 rot1 = mat3(cos(primary), 0.0, sin(primary),
                     0.0, 1.0, 0.0,
                     -sin(primary), 0.0, cos(primary));
    pos = rot1 * pos;
    
    // Secondary rotation.
    float secondary = uTime * uSecondaryRotationSpeed;
    mat3 rot2 = mat3(1.0, 0.0, 0.0,
                     0.0, cos(secondary), -sin(secondary),
                     0.0, sin(secondary), cos(secondary));
    pos = rot2 * pos;
    
    // Differential rotation with chaos.
    float globalRotFactor = 0.8 + 0.2 * sin(uTime * 0.5);
    float diffExponent = 1.0 + 0.5 * sin(uTime * 0.3);
    float diffAngle = uTime * uRotationSpeed * globalRotFactor * pow(1.0 - dist, diffExponent);
    diffAngle += 0.1 * snoise(vec3(pos.xy * 10.0, uTime * 0.2));
    // Twist term.
    float twist = 0.5 * sin(uTime * 1.0 + dist * 5.0);
    diffAngle += twist;
    mat3 diffRot = mat3(cos(diffAngle), 0.0, sin(diffAngle),
                        0.0, 1.0, 0.0,
                        -sin(diffAngle), 0.0, cos(diffAngle));
    pos = diffRot * pos;
    
    // Tilt rotation.
    float tiltAngle = 0.2 * sin(uTime * 1.3 + dist * 3.0);
    mat3 tiltRot = mat3(1.0, 0.0, 0.0,
                        0.0, cos(tiltAngle), -sin(tiltAngle),
                        0.0, sin(tiltAngle), cos(tiltAngle));
    pos = tiltRot * pos;
    
    // Radial drift.
    float radialDrift = 0.002 * sin(uTime * 4.0 + dist * 20.0)
                         + 0.002 * snoise(vec3(pos.yz * 10.0, uTime * 0.3));
    pos += normalize(pos) * radialDrift;
    
    // Global breathing.
    float globalScale = 1.0 + 0.01 * sin(uTime * 0.3);
    pos *= globalScale;
    
    // Vertical warp.
    pos.z += 0.005 * sin(uTime * 2.0 + pos.x * 5.0);
    
    // Additional noise adjustments.
    vec3 extraNoise = vec3(snoise(pos + vec3(uTime * 0.3, 0, 0)),
                           snoise(pos + vec3(0, uTime * 0.3, 0)),
                           snoise(pos + vec3(0, 0, uTime * 0.3)));
    pos += extraNoise * 0.02;
    vec3 nInput = pos * 5.0 + vec3(uTime * 0.1);
    vec3 noiseOff = vec3(snoise(nInput), snoise(nInput + 100.0), snoise(nInput + 200.0));
    pos += noiseOff * 0.03 * (dist + 0.2);
    pos *= sin(uTime * 2.0) * 0.01 + 1.0;
    pos -= normalize(pos) * (0.1 / (dist * dist + 0.1)) * 0.001;
    vec3 radial = normalize(pos);
    vec3 tangent = normalize(cross(vec3(0, 0, 1), radial));
    pos += tangent * 0.005 * sin(uTime * 2.0);
    pos += radial * 0.001 * sin(uTime * 1.5);
    // Extra chaos.
    vec3 chaosOffset = vec3(snoise(vec3(pos.xy * 5.0, uTime)),
                              snoise(vec3(pos.yz * 5.0, uTime)),
                              snoise(vec3(pos.zx * 5.0, uTime)));
    pos += chaosOffset * 0.003;
    // Swarming behavior.
    vec3 swarmOffset = 0.003 * vec3(
      sin(pos.x * 20.0 + uTime * 1.5),
      sin(pos.y * 20.0 + uTime * 1.7),
      sin(pos.z * 20.0 + uTime * 1.3)
    );
    pos += swarmOffset;
    
    // --- CLICK EFFECT: deformation and swarming ---
    if(uClickEffect > 0.001) {
      vec3 toClick = uClickPos - pos;
      float clickDist = length(toClick);
      // The closer the star, the stronger the influence.
      float influence = exp(-clickDist * 3.0) * uClickEffect;
      // Add a swirling (perpendicular) component.
      vec3 perp = normalize(cross(toClick, vec3(0.0, 0.0, 1.0)));
      pos += (toClick + perp * 0.3) * influence;
    }
  }
  gl_Position = uProjectionMatrix * uViewMatrix * vec4(pos, 1.0);
  gl_PointSize = isBlackHole 
    ? uBlackHoleSize / gl_Position.w 
    : mix(12.0, 3.0, dist) * (sin(uTime * 3.0 + dist * 10.0) * 0.2 + 1.5) *
      (sin(uTime * (5.0 + dist * 20.0)) * 0.3 + 1.0) / gl_Position.w;
  vColor = isBlackHole ? aColor : aColor * (0.8 + 0.2 * sin(uTime * 3.0));
  vPrevPos = mix(aPrevPosition, pos, 0.05);
  vCurrentPos = pos;
}
`;

// Simulation Fragment Shader – increased brightness by multiplying the computed color by 2.0.
const simFrag = `#version 300 es
precision highp float;
in vec4 vColor;
in vec3 vCurrentPos, vPrevPos;
out vec4 fragColor;
uniform mat4 uProjectionMatrix, uViewMatrix;
uniform sampler2D uStarTex;
uniform float uTime;
void main(){
  vec2 coord = gl_PointCoord;
  vec2 circCoord = 2.0 * coord - 1.0;
  float d = length(circCoord);
  float twinkle = 0.8 + 0.05 * sin(uTime * 10.0 + coord.x * 10.0);
  if(vColor.rgb == vec3(0.0)){
    float eventHorizon = 0.5;
    float pulsate = 0.5 + 0.5 * sin(uTime * 3.0);
    if(d > eventHorizon){
      float t = (d - eventHorizon) / (1.0 - eventHorizon);
      fragColor = mix(vec4(0.0, 0.0, 0.0, 1.0),
                      vec4(0.2, 0.0, 0.3, 0.0) * pulsate, t);
    } else {
      fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
  } else {
    vec4 texColor = texture(uStarTex, coord);
    float intensity = exp(-2.0 * d * d) * twinkle;
    float halo = smoothstep(0.7, 0.0, d);
    vec3 shiftedColor = vColor.rgb + 0.05 * vec3(sin(uTime + coord.x * 3.14),
                                                  sin(uTime + coord.y * 3.14),
                                                  sin(uTime));
    // Multiply by 2.0 for increased brightness.
    fragColor = vec4((shiftedColor * texColor.rgb * intensity + halo * 0.2) * 2.0,
                     texColor.a * intensity);
  }
}
`;

// Tracer Vertex Shader
const tracerVert = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;
out float vT;
void main(){
  vT = mod(float(gl_VertexID), 2.0);
  gl_Position = uProjectionMatrix * uViewMatrix * vec4(aPos, 1.0);
}
`;

// Tracer Fragment Shader
const tracerFrag = `#version 300 es
precision highp float;
in float vT;
out vec4 fragColor;
void main(){
  vec4 tracerColor = mix(vec4(0.8, 0.8, 1.0, 0.0), vec4(0.8, 0.8, 1.0, 0.6), vT);
  fragColor = tracerColor;
}
`;

// Post Processing Vertex Shader
const postVert = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vTexCoord;
void main(){
  vTexCoord = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// Enhanced Post Processing Fragment Shader (with bloom, lens distortion, vignette, film grain, and high constant contrast)
const postFrag = `#version 300 es
precision highp float;
in vec2 vTexCoord;
uniform sampler2D uScene;
uniform float uTime;
uniform float uDistortionStrength; // Controls lens distortion strength.
uniform float uGrainIntensity;     // Controls film grain intensity.
uniform float uContrast;           // Constant high contrast.
out vec4 fragColor;

//-------------------------------------------------------------
// Utility Functions
//-------------------------------------------------------------
float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898,78.233))) * 43758.5453);
}

//-------------------------------------------------------------
// Bloom Effect: Averages neighboring pixels to add glow.
//-------------------------------------------------------------
vec3 applyBloom(vec2 uv) {
  vec3 baseColor = texture(uScene, uv).rgb;
  float offset = 1.0 / 512.0;
  vec3 bloom = vec3(0.0);
  bloom += texture(uScene, uv + vec2(offset, 0.0)).rgb;
  bloom += texture(uScene, uv - vec2(offset, 0.0)).rgb;
  bloom += texture(uScene, uv + vec2(0.0, offset)).rgb;
  bloom += texture(uScene, uv - vec2(0.0, offset)).rgb;
  bloom += texture(uScene, uv + vec2(offset, offset)).rgb;
  bloom += texture(uScene, uv - vec2(offset, offset)).rgb;
  bloom += texture(uScene, uv + vec2(offset, -offset)).rgb;
  bloom += texture(uScene, uv - vec2(offset, -offset)).rgb;
  bloom /= 8.0;
  return mix(baseColor, baseColor + bloom, 0.5);
}

//-------------------------------------------------------------
// Vignette: Darkens the edges to focus on the center.
//-------------------------------------------------------------
vec3 applyVignette(vec3 color, vec2 uv) {
  vec2 center = vec2(0.5);
  float dist = distance(uv, center);
  float vignette = smoothstep(0.8, 0.4, dist);
  return color * vignette;
}

//-------------------------------------------------------------
// Lens Distortion: Applies a subtle barrel distortion effect.
//-------------------------------------------------------------
vec2 applyLensDistortion(vec2 uv, float strength) {
  vec2 center = vec2(0.5);
  vec2 delta = uv - center;
  float dist = length(delta);
  return center + delta * (1.0 + strength * dist * dist);
}

//-------------------------------------------------------------
// Film Grain: Adds a subtle noise overlay to simulate grain.
//-------------------------------------------------------------
vec3 applyFilmGrain(vec3 color, vec2 uv, float time, float intensity) {
  float noise = rand(uv * time) * intensity;
  return color + noise;
}

//-------------------------------------------------------------
// Main Function
//-------------------------------------------------------------
void main(){
  vec2 distortedUV = applyLensDistortion(vTexCoord, uDistortionStrength);
  vec3 color = applyBloom(distortedUV);
  
  float luminance = dot(color, vec3(0.299, 0.587, 0.114));
  if(luminance > 0.8) {
    vec3 flare = vec3(0.3, 0.2, 0.5) * pow(luminance - 0.8, 2.0);
    color += flare;
  }
  
  color = applyVignette(color, distortedUV);
  
  color.r += 0.05 * sin(uTime + distortedUV.x * 10.0);
  color.g += 0.05 * sin(uTime + distortedUV.y * 10.0);
  color.b += 0.05 * sin(uTime + (distortedUV.x + distortedUV.y) * 5.0);
  
  color = applyFilmGrain(color, distortedUV, uTime, uGrainIntensity);
  
  // Apply high, constant contrast.
  color = ((color - 0.5) * uContrast + 0.5);
  
  fragColor = vec4(color, 1.0);
}
`;

// Nebula Vertex Shader
const nebulaVert = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec4 aColor;
uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;
uniform float uTime;
out vec4 vColor;
void main(){
  vec3 pos = aPosition;
  pos.x += 0.005 * sin(uTime + aPosition.y * 3.0);
  pos.y += 0.005 * cos(uTime + aPosition.x * 3.0);
  gl_Position = uProjectionMatrix * uViewMatrix * vec4(pos, 1.0);
  gl_PointSize = 30.0;
  vColor = aColor;
}
`;

// Nebula Fragment Shader
const nebulaFrag = `#version 300 es
precision highp float;
in vec4 vColor;
uniform float uBurst;
out vec4 fragColor;
void main(){
  vec2 center = vec2(0.5);
  float d = distance(gl_PointCoord, center);
  float alpha = smoothstep(0.6, 0.0, d);
  fragColor = vec4(vColor.rgb * (1.0 + uBurst), vColor.a * alpha);
}
`;

// -------------------------
// FBO and Temporary Texture Setup
// -------------------------
function createFramebuffer(gl, width, height) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const depthBuffer = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthBuffer);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, texture };
}

// -------------------------
// Nebula Particle Generation
// -------------------------
const numNebula = 5000;
const nebulaPositions = [];
const nebulaColors = [];
for (let i = 0; i < numNebula; i++){
  let angle = random(0, 2 * Math.PI);
  let radius = random(0.2, 1.5);
  let x = radius * Math.cos(angle);
  let y = radius * Math.sin(angle);
  let z = random(-0.2, 0.2);
  nebulaPositions.push(x, y, z);
  // Warm, soft colors.
  let rCol = random(0.8, 1.0);
  let gCol = random(0.4, 0.6);
  let bCol = random(0.6, 0.8);
  nebulaColors.push(rCol, gCol, bCol, 0.7);
}

// -------------------------
// Main Function
// -------------------------
function main(){
  const canvas = document.getElementById("canvasWebGL2");
  const gl = canvas.getContext("webgl2");
  if (!gl) { console.error("WebGL2 not supported"); return; }
  resizeCanvas(canvas, gl, width, height);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.enable(gl.DEPTH_TEST);
  
  // Create one offscreen FBO for simulation.
  const simFBOObj = createFramebuffer(gl, width, height);
  // Create a temporary texture to copy simulation results.
  const tempTexture = createTempTexture(gl, width, height);
  
  // Create Simulation Program.
  const simProgram = createProgram(gl, simVert, simFrag);
  if (!simProgram) return;
  gl.useProgram(simProgram);
  const starTex = createStarTexture(gl);
  const starTexLoc = gl.getUniformLocation(simProgram, "uStarTex");
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, starTex);
  gl.uniform1i(starTexLoc, 0);
  
  // Generate Galaxy Data.
  const numPoints = 200000, numArms = 5, armSpread = 0.5,
        diskRadius = 1.0, diskThickness = 0.2, bulgeSize = 0.3;
  const positions = [], prevPositions = [], colors = [];
  for (let i = 0; i < numPoints; i++){
    const r = Math.pow(Math.random(), 0.5) * diskRadius;
    const theta = random(0, 2 * Math.PI);
    const armOffset = (i % numArms) * (2 * Math.PI / numArms);
    const spiralTheta = theta + armOffset + armSpread * Math.sqrt(r);
    const x = r * Math.cos(spiralTheta), y = r * Math.sin(spiralTheta);
    const bulgeTransition = Math.pow(Math.min(1, r / bulgeSize), 2);
    const diskProfile = Math.sqrt(1 - Math.pow(r / diskRadius, 2));
    const heightFactor = mix(1, diskProfile, bulgeTransition);
    const maxHeight = diskThickness * heightFactor;
    let z = r < bulgeSize 
          ? r * Math.cos(Math.acos(2 * Math.random() - 1)) * (bulgeSize - r) / bulgeSize 
          : (Math.random() - 0.5) * maxHeight;
    positions.push(x, y, z);
    prevPositions.push(x, y, z);
    const normDist = Math.sqrt(x*x + y*y + z*z) / diskRadius;
    const brightness = 0.5 + 0.5 * (1 - normDist);
    const hue = 210 + Math.random() * 30, sat = 0.6, light = brightness;
    const [cr, cg, cb] = hslToRgb(hue, sat, light);
    colors.push(cr, cg, cb, 1);
  }
  // Add black hole data.
  positions.push(0, 0, 0);
  prevPositions.push(0, 0, 0);
  colors.push(0, 0, 0, 1);
  
  function createBuffer(data, attrib, size, usage, prog) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), usage);
    const loc = gl.getAttribLocation(prog, attrib);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc);
    return buf;
  }
  const posBuffer = createBuffer(positions, "aPosition", 3, gl.DYNAMIC_DRAW, simProgram);
  const prevPosBuffer = createBuffer(prevPositions, "aPrevPosition", 3, gl.DYNAMIC_DRAW, simProgram);
  const colorBuffer = createBuffer(colors, "aColor", 4, gl.STATIC_DRAW, simProgram);
  
  const timeLoc = gl.getUniformLocation(simProgram, "uTime");
  const rotSpeedLoc = gl.getUniformLocation(simProgram, "uRotationSpeed");
  const primaryRotLoc = gl.getUniformLocation(simProgram, "uPrimaryRotationSpeed");
  const secondaryRotLoc = gl.getUniformLocation(simProgram, "uSecondaryRotationSpeed");
  const projMatLoc = gl.getUniformLocation(simProgram, "uProjectionMatrix");
  const viewMatLoc = gl.getUniformLocation(simProgram, "uViewMatrix");
  const blackHoleSizeLoc = gl.getUniformLocation(simProgram, "uBlackHoleSize");
  // Uniforms for click effect.
  const clickPosLoc = gl.getUniformLocation(simProgram, "uClickPos");
  const clickEffectLoc = gl.getUniformLocation(simProgram, "uClickEffect");
  
  // Setup global projection and view matrices.
  const projection = mat4.create();
  mat4.perspective(projection, Math.PI/4, width/height, 0.1, 100);
  const view = mat4.create();
  
  // Setup Tracer Program.
  const tracerProg = createProgram(gl, tracerVert, tracerFrag);
  const tracerProjLoc = gl.getUniformLocation(tracerProg, "uProjectionMatrix");
  const tracerViewLoc = gl.getUniformLocation(tracerProg, "uViewMatrix");
  const tracerPosLoc = gl.getAttribLocation(tracerProg, "aPos");
  const tracerBuffer = gl.createBuffer();
  
  // Setup Post Processing Program.
  const postProg = createProgram(gl, postVert, postFrag);
  const postSceneLoc = gl.getUniformLocation(postProg, "uScene");
  const postTimeLoc = gl.getUniformLocation(postProg, "uTime");
  const postDistortionLoc = gl.getUniformLocation(postProg, "uDistortionStrength");
  const postGrainLoc = gl.getUniformLocation(postProg, "uGrainIntensity");
  const postContrastLoc = gl.getUniformLocation(postProg, "uContrast");
  
  // Setup Nebula Program.
  const nebulaProg = createProgram(gl, nebulaVert, nebulaFrag);
  const nebulaProjLoc = gl.getUniformLocation(nebulaProg, "uProjectionMatrix");
  const nebulaViewLoc = gl.getUniformLocation(nebulaProg, "uViewMatrix");
  const nebulaTimeLoc = gl.getUniformLocation(nebulaProg, "uTime");
  const nebulaBurstLoc = gl.getUniformLocation(nebulaProg, "uBurst");
  // Create nebula buffers.
  const nebulaPosBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nebulaPosBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nebulaPositions), gl.STATIC_DRAW);
  const nebulaColorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nebulaColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nebulaColors), gl.STATIC_DRAW);
  
  // Full-screen quad for post processing.
  const quadVertices = new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
  ]);
  const quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
  
  // -------------------------
  // Set up mouse event listeners.
  // -------------------------
  window.addEventListener('mousedown', (e) => {
    mouseIsDown = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    // On mousedown, simply update the click position without an immediate effect.
    let pos = getWorldCoord(e.clientX, e.clientY, currentProjection, currentView);
    vec3.copy(clickPos, pos);
  });
  
  window.addEventListener('mousemove', (e) => {
    if (mouseIsDown) {
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      let pos = getWorldCoord(e.clientX, e.clientY, currentProjection, currentView);
      vec3.copy(clickPos, pos);
    }
  });
  
  window.addEventListener('mouseup', (e) => {
    mouseIsDown = false;
  });
  
  // -------------------------
  // Draw Function
  // -------------------------
  let time = 0;
  function draw(){
    // Update global projection and view copies for click unprojection.
    mat4.copy(currentProjection, projection);
    mat4.copy(currentView, view);
    
    // --- Simulation Pass ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, simFBOObj.fbo);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.02, 0.0, 0.04, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    gl.useProgram(simProgram);
    gl.uniform1f(timeLoc, time);
    gl.uniform1f(rotSpeedLoc, 0.5);
    gl.uniform1f(primaryRotLoc, 0.5);
    gl.uniform1f(secondaryRotLoc, 0.8);
    gl.uniformMatrix4fv(projMatLoc, false, projection);
    // Set up a dynamic camera orbit.
    mat4.lookAt(view, [2.5 * Math.cos(time * 0.1), 2.5 * Math.sin(time * 0.1), 1.6],
                     [0, 0, 0],
                     [0, 0, 1]);
    gl.uniformMatrix4fv(viewMatLoc, false, view);
    gl.uniform1f(blackHoleSizeLoc, 120.0);
    // Set the click effect uniforms.
    gl.uniform3fv(clickPosLoc, clickPos);
    gl.uniform1f(clickEffectLoc, clickEffect);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, prevPosBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(positions));
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(positions));
    gl.drawArrays(gl.POINTS, 0, positions.length / 3);
    
    // --- Copy Simulation to Temporary Texture ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, simFBOObj.fbo);
    gl.bindTexture(gl.TEXTURE_2D, tempTexture);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
    
    // --- Post Processing Pass: Render Simulation to Screen ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(postProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tempTexture);
    gl.uniform1i(postSceneLoc, 0);
    gl.uniform1f(postTimeLoc, time);
    // Set post processing parameters.
    gl.uniform1f(postDistortionLoc, 0.15);
    gl.uniform1f(postGrainLoc, 0.05);
    gl.uniform1f(postContrastLoc, 2.0); // Constant high contrast.
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    
    // --- Nebula Pass: Draw Nebula Particles Over the Scene ---
    gl.useProgram(nebulaProg);
    gl.uniformMatrix4fv(nebulaProjLoc, false, projection);
    gl.uniformMatrix4fv(nebulaViewLoc, false, view);
    gl.uniform1f(nebulaTimeLoc, time);
    gl.uniform1f(nebulaBurstLoc, 0.0);
    const nebulaPosLoc = gl.getAttribLocation(nebulaProg, "aPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, nebulaPosBuffer);
    gl.vertexAttribPointer(nebulaPosLoc, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(nebulaPosLoc);
    const nebulaColorLoc = gl.getAttribLocation(nebulaProg, "aColor");
    gl.bindBuffer(gl.ARRAY_BUFFER, nebulaColorBuffer);
    gl.vertexAttribPointer(nebulaColorLoc, 4, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(nebulaColorLoc);
    gl.drawArrays(gl.POINTS, 0, numNebula);
    
    // --- Tracer Pass ---
    const numStars = positions.length / 3;
    const tracerData = new Float32Array(numStars * 6);
    for(let i = 0; i < numStars; i++){
      tracerData[i * 6 + 0] = prevPositions[i * 3 + 0];
      tracerData[i * 6 + 1] = prevPositions[i * 3 + 1];
      tracerData[i * 6 + 2] = prevPositions[i * 3 + 2];
      tracerData[i * 6 + 3] = positions[i * 3 + 0];
      tracerData[i * 6 + 4] = positions[i * 3 + 1];
      tracerData[i * 6 + 5] = positions[i * 3 + 2];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, tracerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tracerData, gl.DYNAMIC_DRAW);
    gl.useProgram(tracerProg);
    gl.uniformMatrix4fv(tracerProjLoc, false, projection);
    gl.uniformMatrix4fv(tracerViewLoc, false, view);
    gl.vertexAttribPointer(tracerPosLoc, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(tracerPosLoc);
    gl.drawArrays(gl.LINES, 0, tracerData.length / 3);
    
    // --- Update State and Loop ---
    // If the mouse is held down, ramp up the effect gradually.
    if (mouseIsDown) {
      clickEffect += 0.01;
    }
    // Decay the click effect more slowly.
    clickEffect = Math.max(0, clickEffect - 0.001);
    time += 0.005;
    requestAnimationFrame(draw);
  }
  draw();
}
main();
