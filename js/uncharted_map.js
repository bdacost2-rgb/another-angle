// Uncharted's map: taste model with uncertainty (fog), uncharted targets, A* routes, checkpoints.
// Moved from the Colab notebook; js/uncharted.js supplies the data and replaces the Python callbacks.
(() => {
// ---------- world: themes on a 2-D map ----------
const DATA = window.COMPASS_DATA || null;
const THEMES_DEMO = [
  {n:"Southern Gothic", x:.17, y:.70, pal:["#3b4a2a","#7a4b2a","#c9b38a","#1d1a14"], place:"the Mississippi Delta", year:1932, words:["spanish moss","a revival tent","rot in the porch beams","heat that hums"]},
  {n:"Film noir", x:.33, y:.28, pal:["#0e0e10","#5a5a5f","#d9d4c7","#2b2e3a"], place:"Los Angeles", year:1947, words:["venetian-blind light","wet asphalt","a phone that won't stop","cigarette smoke"]},
  {n:"Afrofuturism", x:.80, y:.26, pal:["#2a1b4a","#d59a2b","#18a49a","#f2e4c4"], place:"Lagos", year:2140, words:["brass and kente","a choir in orbit","ancestors in the code","solar sails"]},
  {n:"Magical realism", x:.55, y:.60, pal:["#6c8a5b","#e2b45a","#b5523b","#f4e9cf"], place:"a river town", year:1967, words:["yellow butterflies","a grandmother who floats","rain for four years","letters from the dead"]},
  {n:"Pastoral", x:.10, y:.30, pal:["#7a9a5c","#d8cfa2","#9db6c4","#4d5a35"], place:"the Yorkshire moors", year:1890, words:["a stone wall","sheep in fog","a kettle on","the long field"]},
  {n:"Cyberpunk", x:.90, y:.60, pal:["#0b0f2a","#ff3ea5","#28e0ff","#2a2f4a"], place:"Neo-Seoul", year:2077, words:["neon in puddles","a vending-machine shrine","rain on chrome","borrowed memories"]},
  {n:"Surrealism", x:.60, y:.14, pal:["#d9c9a3","#2e5c8a","#b8392a","#efe6d2"], place:"Paris", year:1928, words:["a clock melting","a door in the sea","eyes in the bread","a staircase to nothing"]},
  {n:"Documentary", x:.40, y:.86, pal:["#3a3a3a","#8c8577","#d1cbbd","#5b4a3a"], place:"Harlem", year:1968, words:["a stoop at dusk","hands at work","a protest sign","a kitchen radio"]},
  {n:"Minimalism", x:.74, y:.84, pal:["#e9e6df","#b9b2a6","#2c2c2c","#c8cfd2"], place:"Kyoto", year:1975, words:["one chair","a white wall","a single branch","silence"]},
  {n:"Folk horror", x:.06, y:.90, pal:["#2d2a1e","#8a6d3b","#c1b59a","#5e1f17"], place:"a Cornish village", year:1973, words:["straw figures","a maypole","the wrong hymn","harvest smoke"]},
  {n:"Cosmic", x:.94, y:.10, pal:["#05060f","#3d3a7a","#c7b8ff","#1a1f3d"], place:"the Atacama desert", year:1974, words:["a radio dish","too many stars","a signal answering","the cold"]},
  {n:"Romanticism", x:.26, y:.50, pal:["#3e4c5e","#c69c6d","#e8dcc2","#6b3e3a"], place:"the Lake District", year:1818, words:["a storm over the lake","a lone wanderer","ruins","a letter never sent"]},
];
const THEMES = DATA ? DATA.themes : THEMES_DEMO;
const W = 64, H = 40;                              // grid for the model, fog and search
let rnd = mulberry(7);
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){let u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const WORKS = DATA ? DATA.works : [];
if (!DATA) THEMES.forEach((t,ti)=>{for(let i=0;i<20;i++){const x=t.x+gauss(rnd)*.045, y=t.y+gauss(rnd)*.045; if(x>.01&&x<.99&&y>.01&&y<.99) WORKS.push({x,y,ti});}});
// density of existing works on the grid
const DENS = new Float32Array(W*H);
for (let j=0;j<H;j++) for (let i=0;i<W;i++){ const x=(i+.5)/W, y=(j+.5)/H; let d=0;
  for (const w of WORKS){ const dx=x-w.x, dy=(y-w.y)*.625; d+=Math.exp(-(dx*dx+dy*dy)/(2*.035*.035)); } DENS[j*W+i]=d; }
const DMAX = Math.max(...DENS); for (let k=0;k<DENS.length;k++) DENS[k]/=DMAX;
// wide density: "inside the world" (gaps between clusters count, the empty margins of the map don't)
const DWIDE = new Float32Array(W*H);
for (let j=0;j<H;j++) for (let i=0;i<W;i++){ const x=(i+.5)/W, y=(j+.5)/H; let d=0;
  for (const w of WORKS){ const dx=x-w.x, dy=(y-w.y)*.625; d+=Math.exp(-(dx*dx+dy*dy)/(2*.11*.11)); } DWIDE[j*W+i]=d; }
const DWMAX = Math.max(...DWIDE); for (let k=0;k<DWIDE.length;k++) DWIDE[k]/=DWMAX;

// ---------- Bayesian logistic regression on RBF features (Laplace approximation) ----------
const CX=10, CY=7, D=CX*CY+1, WIDTH=.12;
function phi(x,y){ const f=new Float64Array(D); let k=0;
  for (let b=0;b<CY;b++) for (let a=0;a<CX;a++){ const cx=(a+.5)/CX, cy=(b+.5)/CY, dx=x-cx, dy=(y-cy)*.7; f[k++]=Math.exp(-(dx*dx+dy*dy)/(2*WIDTH*WIDTH)); }
  f[k]=1; return f; }
const PHI = []; for (let j=0;j<H;j++) for (let i=0;i<W;i++) PHI.push(phi((i+.5)/W,(j+.5)/H));
const ALPHA = 1.2;                                 // prior precision
let obs = [];                                      // {x,y,yv,w}
let wMAP = new Float64Array(D), SIG = null, MU = new Float32Array(W*H), SD = new Float32Array(W*H);
const sig = z => 1/(1+Math.exp(-z));
function chol(A,n){ const L=new Float64Array(n*n); for(let i=0;i<n;i++){ for(let j=0;j<=i;j++){ let s=A[i*n+j]; for(let k=0;k<j;k++) s-=L[i*n+k]*L[j*n+k]; L[i*n+j]= i===j? Math.sqrt(Math.max(s,1e-12)) : s/L[j*n+j]; } } return L; }
function cholSolve(L,b,n){ const y=new Float64Array(n); for(let i=0;i<n;i++){ let s=b[i]; for(let k=0;k<i;k++) s-=L[i*n+k]*y[k]; y[i]=s/L[i*n+i]; }
  const x=new Float64Array(n); for(let i=n-1;i>=0;i--){ let s=y[i]; for(let k=i+1;k<n;k++) s-=L[k*n+i]*x[k]; x[i]=s/L[i*n+i]; } return x; }
function fit(){
  const F = obs.map(o=>phi(o.x,o.y));
  let w = new Float64Array(D), Hm;
  for (let it=0; it<12; it++){
    const g = new Float64Array(D); Hm = new Float64Array(D*D);
    for (let d=0; d<D; d++){ g[d]=ALPHA*w[d]; Hm[d*D+d]=ALPHA; }
    obs.forEach((o,k)=>{ const f=F[k]; let z=0; for(let d=0;d<D;d++) z+=w[d]*f[d]; const p=sig(z), r=o.w*(p-o.yv), s=o.w*p*(1-p);
      for(let a=0;a<D;a++){ if(!f[a]) continue; g[a]+=r*f[a]; const sa=s*f[a]; for(let b=0;b<=a;b++) Hm[a*D+b]+=sa*f[b]; } });
    for(let a=0;a<D;a++) for(let b=0;b<a;b++) Hm[b*D+a]=Hm[a*D+b];
    const L=chol(Hm,D), step=cholSolve(L,g,D); let mx=0;
    for(let d=0;d<D;d++){ w[d]-=step[d]; mx=Math.max(mx,Math.abs(step[d])); }
    if (mx<1e-5) break;
  }
  wMAP = w; const L = chol(Hm,D);
  // predictive mean and variance on every cell: var = f^T H^-1 f = |L^-1 f|^2
  for (let c=0;c<W*H;c++){ const f=PHI[c]; let z=0; for(let d=0;d<D;d++) z+=w[d]*f[d];
    const y=new Float64Array(D); let v=0; for(let i=0;i<D;i++){ let s=f[i]; for(let k=0;k<i;k++) s-=L[i*D+k]*y[k]; y[i]=s/L[i*D+i]; v+=y[i]*y[i]; }
    MU[c]=z; SD[c]=Math.sqrt(v); }
  fieldDirty=true;
}
let SD0 = 1; // prior sd scale for normalising fog

// ---------- seeds: themes, here and now, what's on your mind ----------
const state = Object.fromEntries(THEMES.map(t=>[t.n,0]));
state["Southern Gothic"]=1; state["Magical realism"]=1; state["Documentary"]=1; state["Cyberpunk"]=-1;
const chipsEl = document.getElementById('chips');
THEMES.forEach(t=>{ const b=document.createElement('button'); b.className='chip'; b.type='button'; b.id='chip-'+t.n.replace(/\W/g,''); b.textContent=t.n; b.dataset.s=state[t.n];
  b.onclick=()=>{ state[t.n] = state[t.n]===0?1: state[t.n]===1?-1:0; b.dataset.s=state[t.n]; }; chipsEl.appendChild(b); });
const now = new Date();
const hr = now.getHours(), night = hr>=19||hr<6, month = now.toLocaleString('en-US',{month:'long'});
const daypart = night?'night': hr<12?'morning':'afternoon';
document.getElementById('nowtxt').textContent = `${now.toLocaleString('en-US',{weekday:'long'})} ${daypart} in ${month}`;
const MIND_HINTS = {"Southern Gothic":["grandmother","house","south","family","church","secret","inherit"], "Documentary":["politic","protest","news","city","work","justice","community"],
  "Romanticism":["love","longing","heart","letter"], "Pastoral":["home","farm","quiet","nature","field"], "Cosmic":["space","star","alone","universe","future"],
  "Afrofuturism":["future","ancestor","diaspora","black","heritage"], "Film noir":["money","lie","crime","night","city"], "Magical realism":["memory","dream","grandmother","ghost","rain"],
  "Folk horror":["ritual","village","harvest","fear"], "Harlem Renaissance":["jazz","music","harlem","poetry","black"], "Dark academia":["school","library","book","study","college"],
  "Cottagecore":["garden","bake","cozy","cottage","flower"], "Brutalism":["concrete","architecture","building"], "Vaporwave":["mall","internet","90s","80s","glitch"],
  "Solarpunk":["climate","hope","green","solar","community"], "Americana":["road","drive","diner","highway","small town"], "Dystopia":["politic","surveillance","control","government","election"],
  "Fairy tale":["fairy","forest","witch","story","castle"], "Greek myth":["myth","god","ancient","hero","fate"], "Desert Southwest":["arizona","tempe","desert","heat","phoenix","monsoon"],
  "Maritime":["sea","ocean","boat","harbor","lighthouse"], "Urban decay":["abandoned","ruin","decay","empty"], "Coming of age":["leaving home","growing","teen","summer","friend","school"], "Surrealism":["dream","strange","absurd"], "Minimalism":["calm","simple","empty","clean"], "Cyberpunk":["tech","phone","screen","ai","data"]};

function seedObs(){
  obs = [];
  THEMES.forEach(t=>{ const s=state[t.n]; if(!s) return; for(let k=0;k<3;k++) obs.push({x:t.x+gauss(rnd)*.03, y:t.y+gauss(rnd)*.03, yv:s>0?1:0, w:.8}); });
  const mind = document.getElementById('mind').value.toLowerCase();
  THEMES.forEach(t=>{ if ((MIND_HINTS[t.n]||[]).some(h=>mind.includes(h))) obs.push({x:t.x, y:t.y, yv:1, w:.5}); });
  if (night){ ["Film noir","Cosmic"].forEach(n=>{ const t=THEMES.find(q=>q.n===n); obs.push({x:t.x,y:t.y,yv:1,w:.2}); }); }
  if (DATA && DATA.train) DATA.train.forEach(o=>obs.push({...o, itemId:o.id}));   // your Another Angle picks and hearts
  if (typeof journey !== 'undefined') journey.obs.forEach(o=>obs.push(o));   // everything you've answered before
  if (!obs.length) obs.push({x:.5,y:.5,yv:1,w:.1});
}

// ---------- targets: high taste (optimistic) x few existing works ----------
let kappa = 1.0, BETA = 0.9;
function targets(k=3, from=null, minDist=.3){
  const sc = new Float32Array(W*H);
  for (let c=0;c<W*H;c++) sc[c] = sig(MU[c] + BETA*SD[c]) * Math.pow(1-DENS[c], 1.5*kappa) * Math.sqrt(DWIDE[c]);
  const out=[]; const idx=[...sc.keys()].sort((a,b)=>sc[b]-sc[a]);
  for (const c of idx){ const i=c%W, j=(c/W)|0; if (i<2||j<2||i>W-3||j>H-3) continue;
    if (from!=null && Math.hypot((from%W-i)/W, (((from/W)|0)-j)/H*.625) < minDist) continue;
    if (out.every(o=>Math.hypot((o.i-i)/W,(o.j-j)/H*.625)>.2)) out.push({i,j,c,score:sc[c]}); if (out.length===k) break; }
  return out;
}
function startCell(){ // where you are now: the existing work your model likes most
  let best=-1e9, bc=0; for (const w of WORKS){ const c=cellOf(w.x,w.y); if (MU[c]>best){best=MU[c]; bc=c;} } return bc; }
const cellOf=(x,y)=>Math.min(H-1,Math.max(0,(y*H)|0))*W+Math.min(W-1,Math.max(0,(x*W)|0));

// ---------- A* with recorded expansions for animation ----------
function astar(s, g){
  const cost = c => 1 + 3.2*(1-sig(MU[c])) + .8*DENS[c];
  const gS=new Float32Array(W*H).fill(Infinity), came=new Int32Array(W*H).fill(-1), closed=new Uint8Array(W*H);
  const gi=g%W, gj=(g/W)|0, hfn=c=>Math.hypot(c%W-gi, ((c/W)|0)-gj);
  const heap=[[hfn(s),s]]; gS[s]=0; const order=[];
  const push=(e)=>{heap.push(e);let i=heap.length-1;while(i>0){const p=(i-1)>>1;if(heap[p][0]<=heap[i][0])break;[heap[p],heap[i]]=[heap[i],heap[p]];i=p;}};
  const pop=()=>{const top=heap[0],last=heap.pop();if(heap.length){heap[0]=last;let i=0;for(;;){const l=2*i+1,r=l+1;let m=i;if(l<heap.length&&heap[l][0]<heap[m][0])m=l;if(r<heap.length&&heap[r][0]<heap[m][0])m=r;if(m===i)break;[heap[m],heap[i]]=[heap[i],heap[m]];i=m;}}return top;};
  while(heap.length){ const [,c]=pop(); if(closed[c]) continue; closed[c]=1; order.push(c); if(c===g) break;
    const i=c%W, j=(c/W)|0;
    for (let dj=-1;dj<=1;dj++) for (let di=-1;di<=1;di++){ if(!di&&!dj) continue; const ni=i+di, nj=j+dj; if(ni<0||nj<0||ni>=W||nj>=H) continue;
      const n=nj*W+ni; if(closed[n]) continue; const step=Math.hypot(di,dj)*(cost(c)+cost(n))/2, ng=gS[c]+step;
      if (ng<gS[n]){ gS[n]=ng; came[n]=c; push([ng+hfn(n)*1.0,n]); } } }
  const path=[]; let c=g; while(c!==-1){ path.push(c); if(c===s) break; c=came[c]; } path.reverse();
  return {order, path, total:gS[g]};
}

// ---------- drawing: the map as a 3-D terrain ----------
// The same 64 x 40 model grid, lifted into space. Height and warmth are your taste (hills where the
// model thinks you'd love it), and fog floats over whatever it is still unsure about. Drag to turn it,
// move the cursor to clear the fog nearby, click to send a signal across it.
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
let CW=0, CH=0;
function resize(){ const r=cv.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1); CW=r.width; CH=r.height; cv.width=CW*dpr; cv.height=CH*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); draw(); }
let anim = {searches:[], t:0, paths:[], cp:null, start:null, tg:[], arrived:false};
let fieldDirty=true;
const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
const N=W*H, ASPECT=W/H;
// terrain: targets follow the model; the drawn values ease toward them, so the land visibly reshapes when you answer
const HTt=new Float32Array(N), FGt=new Float32Array(N), HEt=new Float32Array(N);
const HT=new Float32Array(N), FG=new Float32Array(N).fill(1), HE=new Float32Array(N);
const PX=new Float32Array(N), PY=new Float32Array(N), PK=new Float32Array(N), LIT=new Float32Array(N), VH=new Float32Array(N);
function terrainTargets(){ for (let c=0;c<N;c++){ const like=sig(MU[c]), fog=Math.min(1,SD[c]/SD0);
    HTt[c]=(like-.5)*.34*(1-.45*fog); FGt[c]=fog; HEt[c]=Math.max(0,like-.45)/.55*(1-.7*fog); } fieldDirty=false; }
const HOME={yaw:-.42, tilt:.98};
const cam={yaw:HOME.yaw, tilt:HOME.tilt, vyaw:0, cy:1, sy:0, ct:1, st:0, S:1};
function project(u,v,h){ const X=(u-.5)*ASPECT, Z=v-.5, x=X*cam.cy-Z*cam.sy, z=X*cam.sy+Z*cam.cy;
  const y=z*cam.ct-h*cam.st, depth=z*cam.st+h*cam.ct, k=1/(1-depth*.42);
  return [CW/2+x*k*cam.S, CH*.5+y*k*cam.S, k]; }
function hAt(u,v){ const fx=Math.min(W-1,Math.max(0,u*W-.5)), fy=Math.min(H-1,Math.max(0,v*H-.5)), i=Math.min(W-2,fx|0), j=Math.min(H-2,fy|0), a=fx-i, b=fy-j, c=j*W+i;
  return (VH[c]*(1-a)+VH[c+1]*a)*(1-b)+(VH[c+W]*(1-a)+VH[c+W+1]*a)*b; }
const cellUV = c => [(c%W+.5)/W, (((c/W)|0)+.5)/H];
const onLand = (u,v,lift=0) => project(u,v,hAt(u,v)+lift);
const PUFF=(()=>{ const c=document.createElement('canvas'); c.width=c.height=48; const g=c.getContext('2d'), r=g.createRadialGradient(24,24,0,24,24,24);
  r.addColorStop(0,'rgba(228,228,234,.85)'); r.addColorStop(.45,'rgba(228,228,234,.32)'); r.addColorStop(1,'rgba(228,228,234,0)'); g.fillStyle=r; g.fillRect(0,0,48,48); return c; })();

// ---------- pointer: turn the map, carry a light over it, send signals ----------
const ptr={x:0,y:0,lx:0,ly:0,on:false,a:0,down:false,drag:false,sx:0,sy:0,pick:-1}; let ripples=[];
WORKS.forEach((w,i)=>{ w.ph=(i*2.399)%6.283; w.ox=0; w.oy=0; });
function ptrPos(e){ const r=cv.getBoundingClientRect(); ptr.x=e.clientX-r.left; ptr.y=e.clientY-r.top; if(!ptr.on){ ptr.lx=ptr.x; ptr.ly=ptr.y; } ptr.on=true; }
cv.addEventListener('pointerdown', e=>{ ptrPos(e); ptr.down=true; ptr.drag=false; ptr.sx=e.clientX; ptr.sy=e.clientY; try{ cv.setPointerCapture(e.pointerId); }catch(_){} });
cv.addEventListener('pointermove', e=>{ const dx=e.clientX-ptr.sx, dy=e.clientY-ptr.sy; ptrPos(e);
  if (ptr.down){ if (!ptr.drag && Math.hypot(dx,dy)>4) ptr.drag=true;
    if (ptr.drag){ cam.yaw+=dx*.006; cam.vyaw=dx*.006; cam.tilt=Math.min(1.3,Math.max(.12,cam.tilt+dy*.004)); ptr.sx=e.clientX; ptr.sy=e.clientY; cv.classList.add('turning'); } } });
function ptrUp(e){ const room = ptr.down && !ptr.drag ? roomAt(ptr.x, ptr.y) : null;
  if (room){ ptr.down=false; ptr.on=false; cv.classList.remove('turning'); enterRoom(room.index); return; }
  if (ptr.down && !ptr.drag && ptr.pick>=0){ const [u,v]=cellUV(ptr.pick); ripples.push({u,v,t:performance.now()}); if (ripples.length>6) ripples.shift(); }
  ptr.down=false; ptr.drag=false; cv.classList.remove('turning'); if (e.pointerType==='touch') ptr.on=false; }
cv.addEventListener('pointerup', ptrUp); cv.addEventListener('pointercancel', ptrUp);
cv.addEventListener('pointerleave', ()=>{ if (!ptr.down) ptr.on=false; });
cv.addEventListener('dblclick', ()=>{ cam.yaw=HOME.yaw; cam.tilt=HOME.tilt; cam.vyaw=0; });
const SANS="'Google Sans Flex', 'Helvetica Neue', Arial, sans-serif";
const SERIF="Newsreader, Georgia, serif";

// ---------- the gallery's rooms on the map, and reading the land ----------
// Each gallery room is drawn as a constellation: its works joined by their shortest connecting tree.
function mst(points){ const n=points.length, inTree=new Uint8Array(n), best=new Float64Array(n).fill(Infinity), from=new Int32Array(n).fill(-1), edges=[];
  if (!n) return edges; best[0]=0;
  for (let it=0; it<n; it++){ let u=-1; for (let i=0;i<n;i++) if(!inTree[i] && (u<0 || best[i]<best[u])) u=i;
    inTree[u]=1; if (from[u]>=0) edges.push([from[u],u]);
    for (let v=0; v<n; v++){ if (inTree[v]) continue; const d=Math.hypot((points[u][0]-points[v][0])*ASPECT, points[u][1]-points[v][1]); if (d<best[v]){ best[v]=d; from[v]=u; } } }
  return edges; }
function segDist(px,py,a,b){ const dx=b[0]-a[0], dy=b[1]-a[1], L=dx*dx+dy*dy||1, t=Math.max(0,Math.min(1,((px-a[0])*dx+(py-a[1])*dy)/L)); return Math.hypot(px-a[0]-t*dx, py-a[1]-t*dy); }
let hoverRoom=null;
function roomAt(x,y){ let best=null, bd=14;
  for (const room of (window.GALLERY_ROOMS||[])){ if (!room.screen) continue;
    room.screen.forEach(p=>{ const d=Math.hypot(p[0]-x,p[1]-y); if (d<bd){ bd=d; best=room; } });
    room.edges.forEach(([a,b])=>{ const d=segDist(x,y,room.screen[a],room.screen[b]); if (d<Math.min(bd,7)){ bd=d; best=room; } }); }
  return best; }
function enterRoom(index){ go('gallery'); requestAnimationFrame(()=>window.dispatchEvent(new CustomEvent('gallery:enter-room',{detail:{index}}))); }
// The card that follows the cursor: what this spot is like (CLIP design dials) and the works nearest to it,
// or, over a constellation, the gallery room it belongs to.
const tipEl=document.getElementById('map-tip'); let tipKey='';
function tipHTML(room, cell){
  const e=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  if (room) return `<small>${room.ring?'final gallery room':'gallery room '+String(room.index+1).padStart(2,'0')}</small><strong>${e(room.title)}</strong>`+
    (room.signature.length?`<span class="map-tip-words">${room.signature.map(e).join(' · ')}</span>`:'')+`<span class="map-tip-cta">${room.points.length} of your works · click to walk in →</span>`;
  const describe=(window.UNCHARTED_BRIDGE||{})['compass.describe']; if (!describe) return '';
  const [u,v]=cellUV(cell), d=describe(u,v);
  const words = d.empty ? '<span class="map-tip-words">uncharted · no works here yet</span>' : d.words.length ? `<span class="map-tip-words">${d.words.map(e).join(' · ')}</span>` : '<span class="map-tip-words">a mix of everything</span>';
  const thumbs = d.works.map(w=>w.img?`<img src="${e(w.img)}" alt="" referrerpolicy="no-referrer" onerror="this.remove()">`:`<span class="map-tip-page">${e(w.title.slice(0,40))}</span>`).join('');
  return `<small>${d.empty?'open ground':'this area'}</small>${words}<div class="map-tip-works">${thumbs}</div><span class="map-tip-names">nearest: ${d.works.map(w=>e(w.title.slice(0,38))).join(' · ')}</span>`;
}
function updateTip(){
  if (!tipEl) return;
  const show = ptr.on && ptr.a>.3 && !ptr.drag && !ptr.down && (hoverRoom || ptr.pick>=0) && DATA;
  if (!show){ if (!tipEl.hidden){ tipEl.hidden=true; tipKey=''; } return; }
  const cell=ptr.pick, key = hoverRoom ? 'r'+hoverRoom.index+hoverRoom.title : 'c'+(((cell/W)|0)>>1)+'_'+((cell%W)>>1);
  if (key!==tipKey){ tipKey=key; tipEl.innerHTML=tipHTML(hoverRoom, cell); tipEl.classList.toggle('is-room', Boolean(hoverRoom)); }
  tipEl.hidden=false;
  const w=tipEl.offsetWidth, h=tipEl.offsetHeight;
  tipEl.style.transform=`translate(${Math.round(Math.min(Math.max(8,ptr.x+22), CW-w-8))}px, ${Math.round(Math.min(Math.max(8,ptr.y+22), CH-h-8))}px)`;
}

function draw(){
  if (!CW || !CH) return;
  const now=performance.now();
  if (fieldDirty) terrainTargets();
  // camera: glide after a drag, and drift very slowly when nobody is touching it
  if (!ptr.down){ cam.yaw+=cam.vyaw; cam.vyaw*=.93; if (!ptr.on && !REDUCE) cam.yaw+=.0007; }
  cam.cy=Math.cos(cam.yaw); cam.sy=Math.sin(cam.yaw); cam.ct=Math.cos(cam.tilt); cam.st=Math.sin(cam.tilt); cam.S=Math.min(CW*.5, CH*.92);
  ctx.clearRect(0,0,CW,CH);
  const bg=ctx.createRadialGradient(CW/2,CH*.5,0,CW/2,CH*.5,Math.max(CW,CH)*.65); bg.addColorStop(0,'#1b1b1f'); bg.addColorStop(1,'#121214'); ctx.fillStyle=bg; ctx.fillRect(0,0,CW,CH);
  ripples=ripples.filter(rp=>now-rp.t<2600);
  ptr.a+=((ptr.on?1:0)-ptr.a)*.08; ptr.lx+=(ptr.x-ptr.lx)*(REDUCE?1:.2); ptr.ly+=(ptr.y-ptr.ly)*(REDUCE?1:.2);
  const R=Math.max(90, Math.min(CW,CH)*.2), ease=REDUCE?1:.07;
  // vertices: ease toward the model, add ripples, project, and find the one under the cursor
  let pick=-1, pd=1e9;
  for (let c=0;c<N;c++){
    HT[c]+=(HTt[c]-HT[c])*ease; FG[c]+=(FGt[c]-FG[c])*ease; HE[c]+=(HEt[c]-HE[c])*ease;
    const u=(c%W+.5)/W, v=(((c/W)|0)+.5)/H; let h=HT[c];
    for (const rp of ripples){ const age=now-rp.t, d=Math.hypot((u-rp.u)*ASPECT, v-rp.v), k=1-Math.abs(d-age*.00055)/.06; if (k>0) h+=k*(1-age/2600)*.05; }
    VH[c]=h; const p=project(u,v,h); PX[c]=p[0]; PY[c]=p[1]; PK[c]=p[2];
    const d=Math.hypot(p[0]-ptr.lx, p[1]-ptr.ly); LIT[c]=d<R ? (1-d/R)*ptr.a : 0; if (d<pd){ pd=d; pick=c; }
  }
  ptr.pick = ptr.a>.01 && pd<60 ? pick : -1;
  // the land: a fine mesh, warm and bright where your taste is
  ctx.lineWidth=1; ctx.strokeStyle='rgba(244,244,244,.055)'; ctx.beginPath();
  for (let j=0;j<H;j+=2){ ctx.moveTo(PX[j*W],PY[j*W]); for (let i=1;i<W;i++) ctx.lineTo(PX[j*W+i],PY[j*W+i]); }
  for (let i=0;i<W;i+=2){ ctx.moveTo(PX[i],PY[i]); for (let j=1;j<H;j++) ctx.lineTo(PX[j*W+i],PY[j*W+i]); }
  ctx.stroke();
  for (let c=0;c<N;c++){ const he=HE[c], l=LIT[c], s=(1.1+he*1.6+l*1.2)*PK[c];
    ctx.fillStyle=`rgba(${150+105*he|0},${150+48*he|0},${165-70*he|0},${Math.min(1,.2+he*.7+l*.5)})`; ctx.fillRect(PX[c]-s/2,PY[c]-s/2,s,s); }
  // works
  let near=null, nd=1e9; const links=[];
  for (const w of WORKS){ const p=onLand(w.x,w.y,.012); let x=p[0], y=p[1], boost=0, tx=0, ty=0;
    if (!REDUCE){ x+=Math.cos(now/1700+w.ph)*.7; y+=Math.sin(now/2100+w.ph*1.7)*.7; }
    if (ptr.a>.01){ const dx=x-ptr.lx, dy=y-ptr.ly, d=Math.hypot(dx,dy);
      if (d<R){ const k=1-d/R; boost=k*ptr.a; const q=k*k*10*ptr.a/(d||1); tx=dx*q; ty=dy*q; if (d<nd){ nd=d; near=w; } if (d<R*.6) links.push([w,d]); } }
    if (REDUCE){ tx=0; ty=0; }
    w.ox+=(tx-w.ox)*.2; w.oy+=(ty-w.oy)*.2; x+=w.ox; y+=w.oy; w.sx=x; w.sy=y;
    ctx.fillStyle = DATA ? (w.mine ? '#F2C14E' : w.kind ? '#b7cbf2' : '#f4f4f4') : THEMES[w.ti].pal[2];
    ctx.globalAlpha=Math.min(1,(w.mine?.95:.72)+boost*.4); ctx.beginPath(); ctx.arc(x,y,(w.mine?2.8:1.7)*p[2]*(1+boost*1.3),0,7); ctx.fill(); }
  ctx.globalAlpha=1;
  // fog: soft puffs floating over what the model is unsure about; the cursor's light burns it off
  for (let c=0;c<N;c++){ const f=Math.max(0,FG[c]-.25)/.75; if (f<=.02) continue; const a=Math.pow(f,1.4)*.2*(1-LIT[c])*(1-LIT[c]); if (a<.006) continue;
    const s=cam.S/W*3.4*PK[c], lift=.05*cam.S*PK[c]*cam.st; ctx.globalAlpha=a; ctx.drawImage(PUFF,PX[c]-s/2,PY[c]-lift-s/2,s,s); }
  ctx.globalAlpha=1;
  if (ptr.a>.01){ const lg=ctx.createRadialGradient(ptr.lx,ptr.ly,0,ptr.lx,ptr.ly,R); lg.addColorStop(0,`rgba(255,229,207,${.13*ptr.a})`); lg.addColorStop(1,'rgba(255,229,207,0)');
    ctx.globalCompositeOperation='lighter'; ctx.fillStyle=lg; ctx.fillRect(ptr.lx-R,ptr.ly-R,2*R,2*R); ctx.globalCompositeOperation='source-over'; }
  // threads from the light to the works closest to it
  if (ptr.a>.01){ links.sort((a,b)=>a[1]-b[1]); ctx.lineWidth=.7;
    for (const [w,d] of links.slice(0,8)){ ctx.strokeStyle=`rgba(244,244,244,${.3*(1-d/(R*.6))*ptr.a})`; ctx.beginPath(); ctx.moveTo(ptr.lx,ptr.ly); ctx.lineTo(w.sx,w.sy); ctx.stroke(); } }
  // signals: rings that travel over the land
  for (const rp of ripples){ const age=now-rp.t, rr=age*.00055; ctx.strokeStyle=`rgba(183,203,242,${.6*(1-age/2600)})`; ctx.lineWidth=1.2; ctx.beginPath();
    let pen=false; for (let k=0;k<=56;k++){ const a=k/56*6.2832, u=rp.u+Math.cos(a)*rr/ASPECT, v=rp.v+Math.sin(a)*rr; if (u<0||u>1||v<0||v>1){ pen=false; continue; }
      const p=onLand(u,v,.004); pen?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); pen=true; }
    ctx.stroke(); }
  // theme names stand on thin stems above the land
  ctx.textAlign='center'; const fs=Math.max(10,Math.min(13,CW/92));
  ctx.font=`500 ${fs}px ${SANS}`;
  // (the ones you chose first; a name that would sit on top of another is left out until you turn the map)
  const placed=[];
  // gallery rooms as constellations (drawn first, so their names win the space)
  const ROOMS = window.GALLERY_ROOMS || [];
  for (const room of ROOMS){ if (!room.edges) room.edges=mst(room.points); room.screen=room.points.map(([x,y])=>onLand(x,y,.014)); }
  hoverRoom = ptr.a>.01 && !ptr.drag ? roomAt(ptr.x, ptr.y) : null;
  for (const room of [...ROOMS].sort((p,q)=>(q.minor?1:0)-(p.minor?1:0))){ if (!room.screen.length) continue; const on=room===hoverRoom, a=on?1:room.minor?.22:.5;
    ctx.strokeStyle=`rgba(255,229,207,${a*.75})`; ctx.lineWidth=on?1.8:1; ctx.setLineDash(room.minor && !on?[2,5]:[]);
    ctx.beginPath(); room.edges.forEach(([i,j])=>{ ctx.moveTo(room.screen[i][0],room.screen[i][1]); ctx.lineTo(room.screen[j][0],room.screen[j][1]); }); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=`rgba(255,229,207,${a})`; room.screen.forEach(p=>{ ctx.beginPath(); ctx.arc(p[0],p[1],(on?3.6:2.6)*p[2],0,7); ctx.fill(); });
    const cx=room.points.reduce((t,p)=>t+p[0],0)/room.points.length, cy=room.points.reduce((t,p)=>t+p[1],0)/room.points.length;
    let li=0; room.points.forEach((p,k)=>{ if (Math.hypot(p[0]-cx,p[1]-cy) < Math.hypot(room.points[li][0]-cx,room.points[li][1]-cy)) li=k; });
    const g=room.screen[li], top=onLand(room.points[li][0], room.points[li][1], .17), label=`${room.ring?'ring':String(room.index+1).padStart(2,'0')}  ${room.title}`;
    ctx.font=`600 ${fs}px ${SANS}`; const tw=ctx.measureText(label).width/2+6;
    if (room.minor && !on) continue;
    if (!on && placed.some(q=>Math.abs(q[0]-top[0])<q[2]+tw && Math.abs(q[1]-top[1])<fs+5)) continue;
    ctx.strokeStyle=`rgba(255,229,207,${a*.5})`; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(g[0],g[1]); ctx.lineTo(top[0],top[1]); ctx.stroke();
    ctx.fillStyle=`rgba(255,229,207,${on?1:.85})`; ctx.fillText(label, top[0], top[1]-5); placed.push([top[0],top[1],tw]); }
  // what the land is like: named areas from the CLIP design dials, lying flat like place names on a map
  ctx.font=`italic 400 ${fs+3}px ${SERIF}`;
  for (const r of (DATA && DATA.regions) || []){ const p=onLand(r.x,r.y,.02), tw=ctx.measureText(r.text).width/2+6;
    if (placed.some(q=>Math.abs(q[0]-p[0])<q[2]+tw && Math.abs(q[1]-p[1])<fs+8)) continue; placed.push([p[0],p[1],tw]);
    ctx.fillStyle='rgba(244,244,244,.4)'; ctx.fillText(r.text, p[0], p[1]); }
  ctx.font=`500 ${fs}px ${SANS}`;
  [...THEMES].sort((a,b)=>(state[b.n]===1)-(state[a.n]===1)).forEach(t=>{ const g=onLand(t.x,t.y,0), p=onLand(t.x,t.y,.1), tw=ctx.measureText(t.n).width/2+6;
    if (placed.some(q=>Math.abs(q[0]-p[0])<q[2]+tw && Math.abs(q[1]-p[1])<fs+5)) return; placed.push([p[0],p[1],tw]);
    ctx.strokeStyle='rgba(244,244,244,.18)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(g[0],g[1]); ctx.lineTo(p[0],p[1]); ctx.stroke();
    ctx.fillStyle = state[t.n]===1?'#f4f4f4': state[t.n]===-1?'rgba(183,203,242,.6)':'rgba(244,244,244,.5)'; ctx.fillText(t.n, p[0], p[1]-5); });
  // your earlier routes and the blocks you worked through
  ctx.setLineDash([2,4]); ctx.lineWidth=1.2; ctx.strokeStyle='rgba(183,203,242,.5)';
  journey.paths.forEach(pth=>{ ctx.beginPath(); pth.forEach(([x,y],k)=>{ const p=onLand(x,y,.006); k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }); ctx.stroke(); });
  ctx.setLineDash([]);
  journey.blocks.forEach(b=>{ const [x,y]=onLand(b.x,b.y,.01); ctx.strokeStyle = b.helped ? '#b7cbf2' : '#9a9a9d'; ctx.lineWidth=1.6;
    ctx.beginPath(); ctx.moveTo(x-4,y-4); ctx.lineTo(x+4,y+4); ctx.moveTo(x+4,y-4); ctx.lineTo(x-4,y+4); ctx.stroke();
    ctx.fillStyle='rgba(244,244,244,.7)'; ctx.font=`500 ${Math.max(9,fs-2)}px ${SANS}`; ctx.fillText(b.block, x, y-8); });
  // explored frontier
  anim.searches.forEach((s)=>{ const n=Math.min(s.order.length, Math.floor(anim.t*s.speed)); const after=Math.max(0, anim.t*s.speed - s.order.length);
    const fade = Math.max(.25, 1 - after/(s.path.length*2+1));
    for (let k=0;k<n;k++){ const c=s.order[k], age=(n-k)/Math.max(1,n), a=(k>n-40? .8 : Math.max(.10,.35-age*.3))*fade, rr=cam.S/W*.3*PK[c]*(k>n-40?1.4:1);
      ctx.fillStyle=`rgba(224,123,76,${a})`; ctx.beginPath(); ctx.arc(PX[c], PY[c], rr, 0, 7); ctx.fill(); } });
  // paths
  anim.searches.forEach((s,si)=>{ const done = anim.t*s.speed >= s.order.length; if(!done) return;
    const prog = Math.min(1, (anim.t*s.speed - s.order.length)/ (s.path.length*1.4));
    const m = Math.max(2, Math.floor(s.path.length*prog));
    ctx.strokeStyle = si===anim.best ? '#E07B4C' : 'rgba(224,123,76,.45)'; ctx.lineWidth = si===anim.best?3:1.6; ctx.setLineDash(si===anim.best?[]:[5,5]);
    ctx.beginPath(); s.path.slice(0,m).forEach((c,k)=>{ k?ctx.lineTo(PX[c],PY[c]):ctx.moveTo(PX[c],PY[c]); }); ctx.stroke(); ctx.setLineDash([]); });
  // beacons: a column of light over each uncharted target, over you, and over the checkpoint
  const beacon=(c,col,hgt)=>{ const [u,v]=cellUV(c), top=project(u,v,VH[c]+hgt), g=ctx.createLinearGradient(PX[c],PY[c],top[0],top[1]); g.addColorStop(0,col); g.addColorStop(1,'rgba(0,0,0,0)');
    ctx.strokeStyle=g; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(PX[c],PY[c]); ctx.lineTo(top[0],top[1]); ctx.stroke(); };
  anim.tg.forEach((t,ti)=>{ const c=t.c!=null?t.c:t.j*W+t.i, x=PX[c], y=PY[c], pulse = 6+2*Math.sin(now/300+ti); beacon(c,'rgba(183,203,242,.9)',.2);
    ctx.strokeStyle='#b7cbf2'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.ellipse(x,y,pulse+4,(pulse+4)*Math.max(.25,cam.ct),0,0,7); ctx.stroke();
    ctx.fillStyle='#b7cbf2'; ctx.beginPath(); ctx.arc(x,y,3.2,0,7); ctx.fill(); });
  if (anim.start!=null){ const c=anim.start, x=PX[c], y=PY[c]; beacon(c,'rgba(244,244,244,.9)',.14); ctx.fillStyle='#f4f4f4'; ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.fill();
    ctx.font=`500 ${fs}px ${SANS}`; ctx.fillText('you', x, y+17); }
  if (anim.cp!=null){ const c=anim.cp, x=PX[c], y=PY[c], p=8+3*Math.sin(now/180); beacon(c,'rgba(255,229,207,1)',.3);
    ctx.strokeStyle='#ffe5cf'; ctx.lineWidth=2; ctx.beginPath(); ctx.ellipse(x,y,p*1.4,p*1.4*Math.max(.25,cam.ct),0,0,7); ctx.stroke(); ctx.fillStyle='#ffe5cf'; ctx.beginPath(); ctx.arc(x,y,3.5,0,7); ctx.fill(); }
  // the cursor itself, and the name of the work it is closest to
  if (ptr.a>.01 && !ptr.drag){ ctx.strokeStyle=`rgba(244,244,244,${.85*ptr.a})`; ctx.lineWidth=1; ctx.beginPath(); ctx.arc(ptr.x,ptr.y,5,0,7); ctx.stroke();
    if (near && nd<18){ const what = DATA ? (near.mine?'your photo':near.kind?'passage':'photo') : THEMES[near.ti].n, label = near.title ? `${what} · ${near.title}` : what;
      ctx.beginPath(); ctx.arc(near.sx,near.sy,7,0,7); ctx.stroke();
      ctx.font=`500 11px ${SANS}`; ctx.textAlign='left';
      const tw=ctx.measureText(label).width, lx=Math.min(near.sx+14, CW-tw-14), ly=Math.max(22, near.sy-12);
      ctx.fillStyle=`rgba(18,18,20,${.82*ptr.a})`; ctx.fillRect(lx-6,ly-13,tw+12,19); ctx.fillStyle=`rgba(244,244,244,${ptr.a})`; ctx.fillText(label,lx,ly); ctx.textAlign='center'; } }
  updateTip();
}

// The website's bridge (js/uncharted.js) does what the Colab Python callbacks used to do.
const BRIDGE = window.UNCHARTED_BRIDGE || {};
async function callPy(name, args){ const f = BRIDGE[name]; if (!f) return null; try { return await f(...args); } catch(e) { console.error(e); return null; } }

// ---------- the loop ----------
const $=id=>document.getElementById(id);
let history=[], reroutes=0, current=null, busy=false;
// ---------- sections: seed, map, checkpoint, logbook. One on screen at a time; the rail switches. ----------
const PANES=['seed','gallery','map','checkpoint','logbook']; let pane=DATA && DATA.trained>=8?'gallery':'seed', paneTimer=0;
// Checkpoints are part of "explore" (the map), so they light up its rail item instead of their own.
const railOf = name => name==='checkpoint' ? 'map' : name;
function go(name){ clearTimeout(paneTimer); pane=name;
  document.querySelector('.shell').classList.toggle('gallery-mode', name==='gallery');
  document.body.classList.toggle('gallery-full', name==='gallery'); if (name!=='gallery') document.body.classList.remove('is-immersed');
  PANES.forEach(n=>{ $('pane-'+n).hidden = n!==name; });
  document.querySelectorAll('.rail [data-pane]').forEach(b=>{ if (b.dataset.pane===railOf(name)){ b.setAttribute('aria-current','true'); b.removeAttribute('data-new'); } else b.removeAttribute('aria-current'); });
  if (name==='gallery' && document.activeElement?.closest('.rail')) $('gallery-stage').focus({preventScroll:true});
  if (name==='gallery') window.dispatchEvent(new CustomEvent('gallery:shown'));
  if (name==='map') requestAnimationFrame(resize); window.scrollTo(0,0); if (name==='checkpoint') queueSpiral(); }
function flag(name){ if (railOf(pane)!==railOf(name)) document.querySelector(`.rail [data-pane="${railOf(name)}"]`).setAttribute('data-new','1'); }
// Once you've picked works in Inspiration, the start step isn't needed: the rail begins at your gallery.
if (DATA && DATA.trained>=8) document.querySelector('.rail [data-pane="seed"]').hidden=true;
let spiralFrame=0;
function queueSpiral(){ if (!spiralFrame) spiralFrame=requestAnimationFrame(updateSpiral); }
// Checkpoint works sit in one straight column. As you scroll, the work in the middle of the screen
// comes into focus like a camera lens pulling focus: sharp, full size and lit, while the ones above and
// below soften, dim and shrink a little. A counter at the side says which one you're on.
function updateSpiral(){
  spiralFrame=0;
  const prog=$('near-progress');
  if (pane!=='checkpoint'){ if (prog) prog.hidden=true; return; }
  const items=[...document.querySelectorAll('#cp .near-item')], vh=window.innerHeight||1;
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let focusI=0, best=Infinity;
  items.forEach((item,i)=>{
    const rect=item.getBoundingClientRect(), off=(rect.top+rect.height/2-vh*.5)/(vh*.6);
    if (Math.abs(off)<best){ best=Math.abs(off); focusI=i; }
    const d=Math.min(1,Math.abs(off)), e=d*d*(3-2*d);
    item.style.setProperty('--focus-scale', reduce?'1':(1-e*.07).toFixed(3));
    item.style.setProperty('--focus-blur', reduce?'0px':`${(e*4.5).toFixed(2)}px`);
    item.style.setProperty('--focus-dim', (1-e*.62).toFixed(3));
  });
  items.forEach((item,i)=>item.classList.toggle('is-focus', i===focusI));
  if (!prog) return;
  const near=document.querySelector('#cp .near'), r=near?.getBoundingClientRect();
  const inView = items.length>1 && r && r.top<vh*.6 && r.bottom>vh*.4;
  prog.hidden=!inView;
  if (!inView) return;
  const pad=n=>String(n).padStart(2,'0'), item=items[focusI];
  prog.querySelector('b').textContent=`${pad(focusI+1)} / ${pad(items.length)}`;
  prog.querySelector('span').textContent=item.classList.contains('near-passage')?'writing':'photograph';
  prog.querySelector('i').replaceChildren(...items.map((_,k)=>{ const t=document.createElement('em'); if (k===focusI) t.className='on'; return t; }));
}
window.addEventListener('scroll', queueSpiral, {passive:true});
window.addEventListener('resize', queueSpiral);
const cpCursor=$('cp-cursor');
document.addEventListener('pointermove', e=>{
  const target=e.target instanceof Element?e.target:null;
  const gallery=target?.closest('#pane-checkpoint .near');
  const item=target?.closest('#pane-checkpoint .near-item');
  if (!gallery || e.pointerType==='touch' || target.closest('button,a,input')){
    cpCursor.classList.remove('is-visible');
    return;
  }
  cpCursor.style.transform=`translate3d(${e.clientX}px,${e.clientY}px,0) translate(-50%,-50%)`;
  const kind=item?.classList.contains('near-passage')?'read':item?'look':'follow';
  cpCursor.dataset.kind=kind;
  cpCursor.firstElementChild.textContent=kind==='read'?'READ':kind==='look'?'LOOK':'FOLLOW';
  cpCursor.classList.add('is-visible');
}, {passive:true});
window.addEventListener('blur', ()=>cpCursor.classList.remove('is-visible'));
document.querySelectorAll('.rail [data-pane]').forEach(b=>b.onclick=()=>go(b.dataset.pane));
document.addEventListener('click', e=>{ const a=e.target.closest('[data-go-pane]'); if (a){ e.preventDefault(); go(a.dataset.goPane); } });
// ---------- your saved identity: answers, routes and blocks carry across sessions ----------
const BLANK = {obs:[], trail:[], paths:[], blocks:[], sessions:0, seenWorkIds:[]};
let journey = (DATA && DATA.journey) || (()=>{ try { return JSON.parse(localStorage.getItem('compass-journey')); } catch(e) { return null; } })() || BLANK;
journey = {...BLANK, ...journey};
if (!Array.isArray(journey.seenWorkIds)) journey.seenWorkIds = [];
function saveJourney(){ try { localStorage.setItem('compass-journey', JSON.stringify(journey)); } catch(e) {} callPy('compass.save', [journey]); }
const cellXY = c => [(c%W+.5)/W, (((c/W)|0)+.5)/H];
function routeFog(path){ if(!path||!path.length) return 1; let m=0; for (const c of path) m+=Math.min(1,SD[c]/SD0); return m/path.length; }
function plan(fromCell){
  anim.start = fromCell; anim.tg = targets(3, fromCell, history.length ? .18 : .3);
  anim.searches = anim.tg.map(t=>{ const r=astar(fromCell, t.c); return {...r, speed: Math.max(20, r.order.length/70)}; });
  // best route: cheapest per step toward the most promising target
  let bi=0, bv=-1e9; anim.searches.forEach((s,i)=>{ const v = anim.tg[i].score*3 - s.total/120; if (v>bv){bv=v; bi=i;} }); anim.best=bi;
  anim.t=0; anim.cp=null; current=null;
  $('sU').textContent = (routeFog(anim.searches[bi].path)*100).toFixed(0)+'%';
}
function checkpointOn(path){ // the foggiest point past the first fifth of the route: where an answer teaches the most
  let best=-1, bc=null; if (path.length < 6) return {cell:null, fog:0};
  path.forEach((c,k)=>{ if (k < path.length*.25 || k > path.length*.85) return; const v=SD[c]/SD0 * (1+.3*k/path.length); if (v>best){best=v; bc=c;} });
  return {cell:bc, fog:best};
}
const FOG_DONE = .42;
function tick(){
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  anim.t += reduce ? 1e6 : 1;
  const all = anim.searches.length && anim.searches.every(s=>anim.t*s.speed >= s.order.length + s.path.length*1.4);
  draw();
  if (all && anim.cp==null && !anim.arrived && busy){
    const s = anim.searches[anim.best]; const cp = checkpointOn(s.path);
    if (!cp.cell || (history.length>=3 && routeFog(s.path) < FOG_DONE) || history.length>=7){ arrive(s); }
    else { anim.cp = cp.cell; ask(cp.cell); }
  }
  requestAnimationFrame(tick);
}
async function ask(cell, label = `Checkpoint ${history.length+1}`, statusText = null){
  $('status').textContent = statusText || `The Compass isn't sure about this area yet (${(Math.min(1,SD[cell]/SD0)*100).toFixed(0)}% unsure). Tell it how these feel.`;
  $('cp').innerHTML = `<div class="tag">${esc(label)}</div><h2 class="thinking">Finding nearby works…</h2>`;
  const nearby = await nearest(cell);
  current = {cell, title: nearby.title};
  $('cp').innerHTML = `<div class="tag">${esc(label)}</div>
    <div class="cpgrid">
    <div class="cp-nearby"><div class="col-h">Proposed works · mark each one</div>${nearby.html || '<p class="imgline">No archive data loaded yet.</p>'}</div>
    <div class="cp-feedback"><div class="col-h">How do you feel about this collection?</div>
    <div class="cp-feedback-actions"><div class="answers">
      <button class="ans" id="a-love" type="button">♥ More like these<kbd>1</kbd></button>
      <button class="ans" id="a-no" type="button">× Not for me<kbd>2</kbd></button>
      <button class="ans" id="a-weird" type="button">Weirder →<kbd>3</kbd></button>
    </div></div></div></div>`;
  showCp();
  $('a-love').onclick=()=>answer('love'); $('a-no').onclick=()=>answer('no'); $('a-weird').onclick=()=>answer('weirder');
}
async function nearest(cell){
  if (!DATA) return {title: "Nearby works", html: ""};
  const x=(cell%W+.5)/W, y=(((cell/W)|0)+.5)/H, d=w=>(w.x-x)**2+(w.y-y)**2;
  const ph = WORKS.filter(w=>!w.kind).sort((a,b)=>d(a)-d(b)).slice(0,2), wr = WORKS.filter(w=>w.kind).sort((a,b)=>d(a)-d(b)).slice(0,1);
  const r = await callPy('compass.peek', [[...ph, ...wr].map(w=>[w.kind, w.id]), journey.seenWorkIds]);
  if (!r) return {title: "Nearby works", html: ""};
  const photos = r.photos || [], passages = r.passages || [];
  const shownIds = [...photos, ...passages].map(p=>p.id).filter(Number.isInteger);
  if (!shownIds.length) return {title: "No new works", html: '<p class="imgline">You have seen all available archive works.</p>'};
  journey.seenWorkIds = [...new Set([...journey.seenWorkIds, ...shownIds])];
  saveJourney();
  const title = [...photos.map(p=>p.title), ...passages.map(p=>p.title)].filter(Boolean).join(" + ").slice(0, 100) || "Nearby works";
  return {
    title,
    html: `<div class="near" aria-label="Nearby archive works">
      ${photos.map((p,i)=>`<figure class="near-item near-photo" data-spiral="${i}"><img src="${esc(p.src)}" alt="${esc(p.title)}" onerror="this.closest('.near-item').remove()"><figcaption class="near-caption"><span class="near-kind">photograph</span><strong class="near-title">${esc(p.title)}</strong><span class="near-credit">${esc(p.credit)}</span></figcaption>${voteBtns(0, p.id)}</figure>`).join('')}
      ${passages.map((p,i)=>`<article class="near-item near-passage" data-spiral="${photos.length+i}"><span class="near-kind">writing</span><h3 class="near-title">${esc(p.title)}</h3><blockquote class="near-quote">${esc(p.text)}</blockquote><p class="near-credit">${esc(p.credit)}</p>${voteBtns(1, p.id)}</article>`).join('')}
    </div>`,
  };
}
function voteBtns(kind, id){ if (id==null) return '';
  return `<div class="vote"><button type="button" class="v" data-k="${kind}" data-i="${id}" data-g="1">♥ Save</button><button type="button" class="v" data-k="${kind}" data-i="${id}" data-g="0">× Less like this</button></div>`; }
// How each answer reads in the logbook (older journeys may also contain "closer").
const ANSWER_LABEL = {love:'♥ more like these', closer:'closer', no:'× not for me', weirder:'weirder →'};
document.addEventListener('click', e=>{ const b = e.target.closest('button.v'); if (!b) return;
  const kind=+b.dataset.k, id=+b.dataset.i, good=b.dataset.g==='1';
  const w = WORKS.find(q=>q.kind===kind && q.id===id); if (!w) return;
  const o = {x:w.x, y:w.y, yv:good?1:0, w:1.0}; obs.push(o); journey.obs.push(o); fit(); saveJourney();
  callPy('compass.vote', [kind, id, good]);
  b.parentElement.querySelectorAll('button.v').forEach(q=>{ q.dataset.on = q===b ? '1' : '0'; q.disabled = true; });
  $('status').textContent = good ? 'Saved. The map warmed around it.' : 'Less like that. That area cools, and it won\'t show again.';
});
function showCp(){ flag('checkpoint'); if (pane==='map'){ clearTimeout(paneTimer); paneTimer=setTimeout(()=>go('checkpoint'), REDUCE?0:2200); } }
function esc(s){ return String(s??'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function answer(a){
  if (!current) return; const c=current.cell, x=(c%W+.5)/W, y=(((c/W)|0)+.5)/H;
  // "More like these" also saves each work in the set you haven't voted on, so they join your gallery
  if (a==='love') document.querySelectorAll('#cp button.v[data-g="1"]:not(:disabled)').forEach(b=>b.click());
  const o = a==='love' ? {x,y,yv:1,w:2.2} : a==='closer' ? {x,y,yv:1,w:.9} : a==='no' ? {x,y,yv:0,w:2.0} : {x,y,yv:1,w:.6};
  obs.push(o); journey.obs.push(o);
  if (a==='weirder'){ kappa += .6; BETA += .4; }
  history.push({title: current.title, ans: a});
  journey.trail.push({title: current.title, ans: a, when: new Date().toISOString().slice(0,10)}); saveJourney();
  const li=document.createElement('li'); li.innerHTML=`<span>${esc(current.title)}</span><span>${esc(ANSWER_LABEL[a]||a)}</span>`; $('trail').appendChild(li); $('trailcard').hidden=false; $('trail-empty').hidden=true;
  $('sQ').textContent = history.length;
  fit(); reroutes++; $('sR').textContent = reroutes;
  const from = (a==='love'||a==='closer') ? c : anim.start;   // move forward only when the step felt right
  $('status').textContent = a==='no' ? 'Dead end. Raising the cost of that area and replanning…' : a==='weirder' ? 'Going stranger: favouring emptier, riskier ground…' : a==='love' ? 'Saved these to your gallery. Moving forward and replanning…' : 'Noted. Moving forward and replanning…';
  $('cp').innerHTML = `<div class="tag">Checkpoint</div><h2>Replanning…</h2><p class="imgline">The model updated. Watch the land reshape and the routes regrow on the map.</p>`;
  current=null; anim.arrived=false; plan(from); go('map');
}
function arrive(s){
  anim.arrived = true; const end = s.path[s.path.length-1];
  journey.paths.push(s.path.filter((c,k)=>k%2===0 || k===s.path.length-1).map(cellXY)); journey.paths = journey.paths.slice(-12); saveJourney(); anim.cp = end;
  const status = `Arrived. Along this route the Compass is now only ${(routeFog(s.path)*100).toFixed(0)}% unsure.`;
  ask(end, "Destination collection", status);
}
function start(){
  journey.sessions = (journey.sessions||0) + 1; saveJourney();
  rnd = mulberry(7 + history.length); history=[]; reroutes=0; kappa=1; BETA=.9; anim.arrived=false;
  $('trail').innerHTML=''; $('trailcard').hidden=true; $('trail-empty').hidden=false; $('sQ').textContent='0'; $('sR').textContent='0';
  seedObs(); fit(); busy=true;
  $('status').textContent='Looking for places your taste points to that nothing in the archive covers yet…';
  $('cp').innerHTML = `<div class="tag">Checkpoint</div><h2>Searching…</h2><p class="imgline">Three routes are heading toward three empty spots your taste points to. Where the Compass is least sure along the way, it stops and brings you here to ask.</p>`;
  plan(startCell()); $('map-cta').hidden=true; go('map');
}
window.addEventListener('compass:gallery-vote', e=>{
  const vote=e.detail;
  if (!vote || vote.id==null || !Number.isFinite(vote.x) || !Number.isFinite(vote.y)) return;
  obs=obs.filter(o=>o.itemId!==vote.id);
  obs.push({...vote, itemId:vote.id});
  fit();
  if (busy){ anim.arrived=false; plan(anim.start ?? startCell()); $('status').textContent='Your gallery choice updated the model. Replanning the routes…'; }
});
document.addEventListener('keydown', e=>{ if (e.target.tagName==='INPUT') return; const m={'1':'love','2':'no','3':'weirder'}[e.key]; if (m && current && pane==='checkpoint') answer(m); });
$('go').onclick=start; $('go-map').onclick=start;

// initial resting frame: prior fog with the example seed applied, no search yet
obs=[]; fit(); SD0 = Math.max(...SD); seedObs(); fit();
$('trail-empty').hidden = journey.trail.length>0;
if (journey.trail.length){ journey.trail.slice(-5).forEach(t=>{ const li=document.createElement('li'); li.innerHTML=`<span>${esc(t.title)}</span><span>${esc(ANSWER_LABEL[t.ans]||t.ans)} · ${esc(t.when||'')}</span>`; $('trail').appendChild(li); }); $('trailcard').hidden=false; }
if (DATA){ $('kindlegend').hidden=false; if (DATA.trained) $('status').textContent = `Built from your ${DATA.trained} picks. Warm high ground is what you love; fog is where the Compass isn't sure. Press “Find new directions” and it heads for places nothing in the archive covers yet, stopping in the fog to ask you.`; }
if (journey.obs.length) $('status').textContent = `Welcome back. Your map remembers ${journey.obs.length} answer${journey.obs.length===1?"":"s"} and ${journey.blocks.length} block${journey.blocks.length===1?"":"s"} from ${journey.sessions} session${journey.sessions===1?'':'s'}.`;
go(pane);
window.addEventListener('resize', resize); if (window.ResizeObserver) new ResizeObserver(()=>{ if (Math.abs(cv.getBoundingClientRect().width-CW)>1) resize(); }).observe(cv); resize(); requestAnimationFrame(tick);
})();
