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
  if (DATA && DATA.train) DATA.train.forEach(o=>obs.push(o));   // your Another Angle picks and hearts
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

// ---------- drawing ----------
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
const off=document.createElement('canvas'); off.width=W; off.height=H; const octx=off.getContext('2d');
let CW=0, CH=0;
function resize(){ const r=cv.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1); CW=r.width; CH=r.width*10/16; cv.width=CW*dpr; cv.height=CH*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); draw(); }
const px=(i)=> (i+.5)/W*CW, py=(j)=> (j+.5)/H*CH;
let anim = {searches:[], t:0, paths:[], cp:null, start:null, tg:[], arrived:false};
let fieldCache=null, fieldDirty=true;
const NOISE=(()=>{ const c=document.createElement('canvas'); c.width=320; c.height=200; const g=c.getContext('2d'); const r=mulberry(99);
  // value noise, three octaves
  const grid=(n)=>{ const a=[]; for(let k=0;k<(n+1)*(n+1);k++) a.push(r()); return a; };
  const oct=[[6,.5],[14,.3],[34,.2]].map(([n,w])=>({n,w,g:grid(n)}));
  const img=g.createImageData(320,200);
  for(let y=0;y<200;y++) for(let x=0;x<320;x++){ let v=0; for(const o of oct){ const fx=x/320*o.n, fy=y/200*o.n, ix=fx|0, iy=fy|0, tx=fx-ix, ty=fy-iy, sm=t=>t*t*(3-2*t);
      const G=(i,j)=>o.g[j*(o.n+1)+i]; const a=G(ix,iy)+(G(ix+1,iy)-G(ix,iy))*sm(tx), b=G(ix,iy+1)+(G(ix+1,iy+1)-G(ix,iy+1))*sm(tx); v+=o.w*(a+(b-a)*sm(ty)); }
    const k=(y*320+x)*4; img.data[k]=img.data[k+1]=img.data[k+2]=255; img.data[k+3]=Math.min(255, 60+v*230); }
  g.putImageData(img,0,0); return c; })();
function buildField(){
  fieldCache = fieldCache || document.createElement('canvas'); fieldCache.width=Math.max(1,CW|0); fieldCache.height=Math.max(1,CH|0);
  const f=fieldCache.getContext('2d');
  // ground: dark, warming where the model is confident you like it
  const img=octx.createImageData(W,H);
  for (let c=0;c<W*H;c++){ const like=sig(MU[c]), fog=Math.min(1,SD[c]/SD0), heat=Math.max(0,like-.45)/.55*(1-.7*fog);
    const k=c*4; img.data[k]=20+105*heat; img.data[k+1]=18+52*heat; img.data[k+2]=16+22*heat; img.data[k+3]=255; }
  octx.putImageData(img,0,0); f.imageSmoothingEnabled=true; f.imageSmoothingQuality='high'; f.drawImage(off,0,0,CW,CH);
  // fog: whitish, textured, thick where the model is unsure
  const t=document.createElement('canvas'); t.width=CW|0; t.height=CH|0; const tg=t.getContext('2d');
  const fimg=octx.createImageData(W,H);
  for (let c=0;c<W*H;c++){ const fog=Math.min(1,SD[c]/SD0); const k=c*4; fimg.data[k]=225; fimg.data[k+1]=221; fimg.data[k+2]=214; fimg.data[k+3]=255*Math.pow(Math.max(0,fog-.25)/.75,1.4)*.55; }
  octx.putImageData(fimg,0,0); tg.imageSmoothingEnabled=true; tg.imageSmoothingQuality='high'; tg.drawImage(off,0,0,CW,CH);
  tg.globalCompositeOperation='destination-in'; tg.drawImage(NOISE,0,0,CW,CH);
  f.drawImage(t,0,0); fieldDirty=false;
}
function drawField(){ if (fieldDirty || !fieldCache || fieldCache.width!==(CW|0)) buildField(); ctx.drawImage(fieldCache,0,0,CW,CH); }
function draw(){
  if (!CW) return;
  ctx.clearRect(0,0,CW,CH); drawField();
  // graticule
  ctx.strokeStyle='rgba(239,232,220,.05)'; ctx.lineWidth=1;
  for(let a=1;a<8;a++){ ctx.beginPath(); ctx.moveTo(a*CW/8,0); ctx.lineTo(a*CW/8,CH); ctx.stroke(); }
  for(let a=1;a<5;a++){ ctx.beginPath(); ctx.moveTo(0,a*CH/5); ctx.lineTo(CW,a*CH/5); ctx.stroke(); }
  // works
  for (const w of WORKS){ ctx.fillStyle = DATA ? (w.mine ? '#F2C14E' : w.kind ? '#86AFC2' : '#E9DFCF') : THEMES[w.ti].pal[2]; ctx.globalAlpha=w.mine?.95:.75; ctx.beginPath(); ctx.arc(w.x*CW,w.y*CH,w.mine?3:1.8,0,7); ctx.fill(); }
  ctx.globalAlpha=1;
  // theme labels
  ctx.textAlign='center'; const fs=Math.max(11,CW/62);
  ctx.font=`italic 500 ${fs}px 'Cormorant Garamond', Georgia, serif`;
  THEMES.forEach(t=>{ ctx.fillStyle = state[t.n]===1?'#EFE8DC': state[t.n]===-1?'rgba(134,175,194,.7)':'rgba(239,232,220,.55)'; ctx.fillText(t.n, t.x*CW, t.y*CH - fs*.9); });
  // your earlier routes and the blocks you worked through
  ctx.setLineDash([2,4]); ctx.lineWidth=1.2; ctx.strokeStyle='rgba(134,175,194,.55)';
  journey.paths.forEach(pth=>{ ctx.beginPath(); pth.forEach(([x,y],k)=> k?ctx.lineTo(x*CW,y*CH):ctx.moveTo(x*CW,y*CH)); ctx.stroke(); });
  ctx.setLineDash([]);
  journey.blocks.forEach(b=>{ const x=b.x*CW, y=b.y*CH; ctx.strokeStyle = b.helped ? '#86AFC2' : '#9A9085'; ctx.lineWidth=1.6;
    ctx.beginPath(); ctx.moveTo(x-4,y-4); ctx.lineTo(x+4,y+4); ctx.moveTo(x+4,y-4); ctx.lineTo(x-4,y+4); ctx.stroke();
    ctx.fillStyle='rgba(239,232,220,.7)'; ctx.font=`500 ${Math.max(9,CW/95)}px 'Instrument Sans', sans-serif`; ctx.fillText(b.block, x, y-8); });
  // explored frontier
  const rr = CW/W*.32;
  anim.searches.forEach((s)=>{ const n=Math.min(s.order.length, Math.floor(anim.t*s.speed)); const after=Math.max(0, anim.t*s.speed - s.order.length);
    const fade = Math.max(.25, 1 - after/(s.path.length*2+1));
    for (let k=0;k<n;k++){ const c=s.order[k], age=(n-k)/Math.max(1,n), a=(k>n-40? .75 : Math.max(.10,.35-age*.3))*fade;
      ctx.fillStyle=`rgba(224,123,76,${a})`; ctx.beginPath(); ctx.arc(px(c%W), py((c/W)|0), k>n-40? rr*1.3 : rr, 0, 7); ctx.fill(); } });
  // paths
  anim.searches.forEach((s,si)=>{ const done = anim.t*s.speed >= s.order.length; if(!done) return;
    const prog = Math.min(1, (anim.t*s.speed - s.order.length)/ (s.path.length*1.4));
    const m = Math.max(2, Math.floor(s.path.length*prog));
    ctx.strokeStyle = si===anim.best ? '#E07B4C' : 'rgba(224,123,76,.45)'; ctx.lineWidth = si===anim.best?3:1.6; ctx.setLineDash(si===anim.best?[]:[5,5]);
    ctx.beginPath(); s.path.slice(0,m).forEach((c,k)=>{ const x=px(c%W), y=py((c/W)|0); k?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); ctx.setLineDash([]); });
  // targets
  anim.tg.forEach((t,ti)=>{ const x=px(t.i), y=py(t.j), pulse = 6+2*Math.sin(performance.now()/300+ti);
    ctx.strokeStyle='#86AFC2'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(x,y,pulse+4,0,7); ctx.stroke();
    ctx.fillStyle='#86AFC2'; ctx.beginPath(); ctx.arc(x,y,3.2,0,7); ctx.fill(); });
  // start
  if (anim.start!=null){ const x=px(anim.start%W), y=py((anim.start/W)|0); ctx.fillStyle='#EFE8DC'; ctx.beginPath(); ctx.arc(x,y,5.5,0,7); ctx.fill();
    ctx.font=`500 ${Math.max(10,CW/80)}px 'Instrument Sans', sans-serif`; ctx.fillText('you', x, y+16); }
  // checkpoint
  if (anim.cp!=null){ const x=px(anim.cp%W), y=py((anim.cp/W)|0), p=8+3*Math.sin(performance.now()/180);
    ctx.strokeStyle='#EFE8DC'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(x-p,y); ctx.lineTo(x+p,y); ctx.moveTo(x,y-p); ctx.lineTo(x,y+p); ctx.stroke();
    ctx.beginPath(); ctx.arc(x,y,p*.7,0,7); ctx.stroke(); }
}

// ---------- blending: what lives at a point ----------
function blendAt(c){ const x=(c%W+.5)/W, y=(((c/W)|0)+.5)/H;
  const ws = THEMES.map(t=>Math.exp(-((x-t.x)**2+((y-t.y)*.625)**2)/(2*.12*.12)));
  const s = ws.reduce((a,b)=>a+b,0)||1; return THEMES.map((t,i)=>({t, w:ws[i]/s})).sort((a,b)=>b.w-a.w).slice(0,3); }
function settingFor(bl){ const a=bl[0].t, b=bl[1].t; return `${a.place[0].toUpperCase()+a.place.slice(1)}, ${b.year < 0 ? (-b.year)+" BCE" : b.year}`; }

// ---------- what could exist here: a blend of the two closest real photos (free, no image generator) ----------
let lastPhotos = [];
function paint(canvas, bl, seed){
  const srcs = lastPhotos.map(p=>p.src).slice(0,2);
  if (!srcs.length) { sketch(canvas, bl, seed); return 'Sketch drawn by code from this spot\'s blend of themes'; }
  const w=canvas.width=480, h=canvas.height=360, g=canvas.getContext('2d');
  const pal=[...bl[0].t.pal]; g.fillStyle=pal[3]; g.fillRect(0,0,w,h);
  const draw1=(img, alpha, mode)=>{ const s=Math.max(w/img.width, h/img.height), iw=img.width*s, ih=img.height*s;
    g.globalAlpha=alpha; g.globalCompositeOperation=mode; g.drawImage(img,(w-iw)/2,(h-ih)/2,iw,ih); };
  const load=src=>new Promise(r=>{ const im=new Image(); im.onload=()=>r(im); im.onerror=()=>r(null); im.src=src; });
  Promise.all(srcs.map(load)).then(([a,b])=>{
    if (!a && !b){ sketch(canvas, bl, seed); return; }
    g.clearRect(0,0,w,h); if (a) draw1(a,1,'source-over'); if (b) draw1(b,.55, a?'soft-light':'source-over');
    g.globalCompositeOperation='color'; g.globalAlpha=.22; g.fillStyle=pal[1]; g.fillRect(0,0,w,h);
    g.globalCompositeOperation='source-over'; g.globalAlpha=1; });
  return 'A blend of the two closest real works: a place between them that nobody has photographed yet';
}

// ---------- procedural sketch (fallback when no photos are loaded) ----------
function sketch(canvas, bl, seed){
  const r=mulberry(seed), w=canvas.width=480, h=canvas.height=360, g=canvas.getContext('2d');
  const pal = [...bl[0].t.pal, ...bl[1].t.pal.slice(0,2)];
  const grd=g.createLinearGradient(0,0,0,h); grd.addColorStop(0,pal[3]); grd.addColorStop(.6,pal[0]); grd.addColorStop(1,pal[1]); g.fillStyle=grd; g.fillRect(0,0,w,h);
  const horizon = h*(.45+r()*.25);
  // distant forms
  for (let k=0;k<5;k++){ g.fillStyle=pal[(k+2)%pal.length]; g.globalAlpha=.25+r()*.4; g.beginPath(); const cx=r()*w, rw=40+r()*160, rh=30+r()*120;
    if (r()<.5){ g.ellipse(cx,horizon-rh*.3,rw,rh,0,0,7); } else { g.rect(cx-rw/4,horizon-rh,rw/2,rh); g.moveTo(cx-rw/4,horizon-rh); g.arc(cx,horizon-rh,rw/4,Math.PI,0); }
    g.fill(); }
  g.globalAlpha=1; g.fillStyle=pal[1]; g.globalAlpha=.85; g.fillRect(0,horizon,w,h-horizon);
  // a light source
  g.globalAlpha=.9; const lx=r()*w, ly=h*(.12+r()*.25); const rg=g.createRadialGradient(lx,ly,2,lx,ly,90); rg.addColorStop(0,pal[2]); rg.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=rg; g.fillRect(0,0,w,h);
  // a figure
  g.globalAlpha=.95; g.fillStyle=pal[3]; const fx=w*(.25+r()*.5), fh=50+r()*40; g.beginPath(); g.ellipse(fx,horizon-fh,7,9,0,0,7); g.fill(); g.fillRect(fx-6,horizon-fh+8,12,fh-8);
  // reflections / lines
  g.strokeStyle=pal[2]; g.globalAlpha=.25; for(let k=0;k<14;k++){ const y=horizon+6+r()*(h-horizon); g.beginPath(); g.moveTo(r()*w,y); g.lineTo(r()*w,y); g.stroke(); }
  // grain
  const id=g.getImageData(0,0,w,h); for(let k=0;k<id.data.length;k+=4){ const n=(r()-.5)*28; id.data[k]+=n; id.data[k+1]+=n; id.data[k+2]+=n; } g.putImageData(id,0,0);
  g.globalAlpha=1;
}

// ---------- text for a checkpoint (Claude if allowed, else composed from the blend) ----------
let samplePromise = (window.claude && window.claude.use) ? window.claude.use("sample") : Promise.resolve(null);
function composed(bl, setting){ const a=bl[0].t, b=bl[1].t, pick=(arr,s)=>arr[Math.floor(mulberry(s)()*arr.length)];
  const s = (obs.length*31)|0;
  return { title: `${a.n} × ${b.n}`, setting,
    image: `A photograph of ${pick(a.words,s)} beside ${pick(b.words,s+1)}, lit like ${night?'a late night':'late afternoon'}.`,
    line: `In ${setting}, ${pick(a.words,s+2)} and ${pick(b.words,s+3)} share the same room, and nobody in the house thinks that is strange.` }; }
// The website's bridge (js/uncharted.js) does what the Colab Python callbacks used to do.
const BRIDGE = window.UNCHARTED_BRIDGE || {};
async function callPy(name, args){ const f = BRIDGE[name]; if (!f) return null; try { return await f(...args); } catch(e) { console.error(e); return null; } }
function parseJSON(t){ try { const m = String(t).match(/\{[\s\S]*\}/); return m ? JSON.parse(m[0]) : null; } catch(e) { return null; } }
function promptFor(bl, setting, history){
  return `You are the imagination inside Aesthetic Compass, a tool that finds the creative work only one person would make.
At this point on their taste map the themes blend as: ${bl.map(b=>`${b.t.n} ${(b.w*100).toFixed(0)}%`).join(', ')}.
The place and time is a blend that never existed: "${setting}".
They are in ${document.getElementById('here').value} on a ${document.getElementById('nowtxt').textContent}. On their mind: "${document.getElementById('mind').value}".
Their reactions so far: ${history.length? history.map(h=>`${h.title}: ${h.ans}`).join('; ') : 'none yet'}.
Invent something strange and specific, never generic. Return JSON only: {"title": "evocative 2-5 word title", "setting": "place, year (may refine the given one)", "image": "one sentence describing a photograph that exists only here", "line": "two sentences of prose in that world, literary, concrete, under 45 words"}`;
}
async function words(bl, setting, history){
  const fallback = composed(bl, setting);
  const sample = await samplePromise.catch(()=>null);
  if (!sample) return fallback;
  const prompt = promptFor(bl, setting, history);
  try { const j = await sample.json(prompt, {modelTier:"quick"}); if (j && j.title && j.line) return j; } catch(e) {}
  return fallback;
}

// ---------- the loop ----------
const $=id=>document.getElementById(id);
let history=[], reroutes=0, current=null, busy=false;
// ---------- your saved identity: answers, routes and blocks carry across sessions ----------
const BLANK = {obs:[], trail:[], paths:[], blocks:[], sessions:0};
let journey = (DATA && DATA.journey) || (()=>{ try { return JSON.parse(localStorage.getItem('compass-journey')); } catch(e) { return null; } })() || BLANK;
journey = {...BLANK, ...journey};
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
async function ask(cell){
  const bl = blendAt(cell), setting = settingFor(bl);
  $('status').textContent = `Checkpoint: the route is foggy here (${(Math.min(1,SD[cell]/SD0)*100).toFixed(0)}% unsure). Asking you.`;
  $('cp').innerHTML = `<div class="tag">Checkpoint ${history.length+1} · ${bl.map(b=>b.t.n+' '+(b.w*100).toFixed(0)+'%').join(' · ')}</div>
    <h2 class="thinking">Imagining this spot…</h2>`;
  const [j, near] = await Promise.all([words(bl, setting, history), nearest(cell)]);
  current = {cell, bl, ...j};
  $('cp').innerHTML = `<div class="tag">Checkpoint ${history.length+1} · ${bl.map(b=>b.t.n+' '+(b.w*100).toFixed(0)+'%').join(' · ')}</div>
    <div class="cpgrid">
    <div><div class="col-h">What the Compass imagines here</div>
      <h2>${esc(j.title)}</h2><div class="setting">${esc(j.setting||setting)}</div>
      <canvas class="sketch" id="sk"></canvas><div class="cap" id="skcap"></div>
      <p class="frag">“${esc(j.line)}”</p><p class="imgline">${esc(j.image)}</p></div>
    <div><div class="col-h">What already exists nearby · mark each one</div>${near || '<p class="imgline">No archive data loaded yet.</p>'}</div>
    <div><div class="col-h">How do you feel about this spot?</div>
    <div class="answers">
      <button class="ans" id="a-love" type="button">Love it<kbd>1</kbd></button>
      <button class="ans" id="a-closer" type="button">Closer<kbd>2</kbd></button>
      <button class="ans" id="a-no" type="button">Not me<kbd>3</kbd></button>
      <button class="ans" id="a-weird" type="button">Weirder<kbd>4</kbd></button>
      <button class="ans stuck" id="a-stuck" type="button">I'm stuck here</button>
    </div><div id="coachbox"></div></div></div>`;
  showCp();
  $('skcap').textContent = paint($('sk'), bl, cell*7+history.length);
  $('a-love').onclick=()=>answer('love'); $('a-closer').onclick=()=>answer('closer'); $('a-no').onclick=()=>answer('no'); $('a-weird').onclick=()=>answer('weirder'); $('a-stuck').onclick=stuck;
}
async function nearest(cell){
  if (!DATA) return '';
  const x=(cell%W+.5)/W, y=(((cell/W)|0)+.5)/H, d=w=>(w.x-x)**2+(w.y-y)**2;
  const ph = WORKS.filter(w=>!w.kind).sort((a,b)=>d(a)-d(b)).slice(0,2), wr = WORKS.filter(w=>w.kind).sort((a,b)=>d(a)-d(b)).slice(0,1);
  const r = await callPy('compass.peek', [[...ph, ...wr].map(w=>[w.kind, w.id])]);
  lastPhotos = (r && r.photos) || [];
  if (!r) return '';
  return `<div class="near">${(r.photos||[]).map(p=>`<figure><img src="${p.src}" alt=""><figcaption>${esc(p.credit)}</figcaption>${voteBtns(0, p.id)}</figure>`).join('')}</div>
    ${(r.passages||[]).map(p=>`<p class="imgline" style="font-family:var(--display);font-size:16px;font-style:italic">${esc(p.text)}</p><div class="cap">${esc(p.credit)}</div>${voteBtns(1, p.id)}`).join('')}`;
}
const BLOCKS = {motivation:"I can't get started", inspiration:"I don't know what to make", confidence:"I'm worried it won't be good"};
function stuck(){
  if (!current) return;
  $('coachbox').innerHTML = `<div class="coach"><div class="tag">Coach · what's in the way?</div>
    <div class="chips" style="margin-top:8px">${Object.entries(BLOCKS).map(([k,v])=>`<button class="chip" type="button" data-b="${k}" data-s="0">${v}</button>`).join('')}</div>
    <input type="text" id="blocktxt" placeholder="Or say it in your own words" style="margin-top:8px">
    <button class="btn" id="coachgo" type="button" style="margin-top:8px;width:100%">Coach me</button><div id="coachout"></div></div>`;
  let chosen = '';
  document.querySelectorAll('#coachbox .chip').forEach(b=>b.onclick=()=>{ chosen=b.dataset.b; document.querySelectorAll('#coachbox .chip').forEach(q=>q.dataset.s = q===b ? '1' : '0'); });
  $('coachgo').onclick = async ()=>{
    const cur = current; if (!cur) return;
    $('coachout').innerHTML = '<p class="thinking">The coach is reading your map…</p>';
    let r = await callPy('compass.coach', [chosen, $('blocktxt').value, cur.title, cur.setting||'', history.map(h=>h.title+': '+h.ans).join('; ')]);
    if (!r || !r.text){ const b = chosen || 'inspiration';
      const local = {motivation:`Set a 10-minute timer. Make three rough frames or three sentences of "${cur.title}". Stop when it rings.`,
        inspiration:`Borrow one detail from this checkpoint (${(cur.image||'').toLowerCase()}) and find its echo where you are today.`,
        confidence:`Make it badly on purpose. Five frames or one paragraph of "${cur.title}" that nobody will see.`};
      r = {text: local[b], block: b, intervention: 'local', label: BLOCKS[b]}; }
    $('coachout').innerHTML = `<p class="imgline" style="color:var(--ink);font-size:15px;margin-top:12px">${esc(r.text)}</p>
      <div class="answers"><button class="ans" id="c-yes" type="button">That helped</button><button class="ans" id="c-no" type="button">Didn't help</button></div>
      <p class="cap" style="margin-top:8px"><a href="index.html#coach" target="_blank" style="color:var(--sea)">Talk it through with the full coach</a></p>`;
    const done = (helped)=>{ const [x,y] = cellXY(cur.cell);
      journey.blocks.push({x, y, block: r.block, helped, intervention: r.intervention}); journey.blocks = journey.blocks.slice(-30);
      if (helped){ const o={x,y,yv:1,w:.6}; obs.push(o); journey.obs.push(o); fit(); } else { kappa = Math.max(.6, kappa-.2); }
      callPy('compass.coach_feedback', [r.block, r.intervention, helped]); saveJourney();
      $('coachbox').innerHTML = `<p class="cap" style="margin-top:10px">Saved to your map. ${helped ? 'This spot now feels more reachable.' : 'The coach will try a different strategy next time, and routes will play it safer.'} Answer the checkpoint when you're ready.</p>`; };
    $('c-yes').onclick=()=>done(true); $('c-no').onclick=()=>done(false);
  };
}
function voteBtns(kind, id){ if (id==null) return '';
  return `<div class="vote"><button type="button" class="v" data-k="${kind}" data-i="${id}" data-g="1">On track</button><button type="button" class="v" data-k="${kind}" data-i="${id}" data-g="0">Off track</button></div>`; }
document.addEventListener('click', e=>{ const b = e.target.closest('button.v'); if (!b) return;
  const kind=+b.dataset.k, id=+b.dataset.i, good=b.dataset.g==='1';
  const w = WORKS.find(q=>q.kind===kind && q.id===id); if (!w) return;
  const o = {x:w.x, y:w.y, yv:good?1:0, w:1.0}; obs.push(o); journey.obs.push(o); fit(); saveJourney();
  callPy('compass.vote', [kind, id, good]);
  b.parentElement.querySelectorAll('button.v').forEach(q=>{ q.dataset.on = q===b ? '1' : '0'; q.disabled = true; });
  $('status').textContent = good ? 'Noted: that one is on track. The map warmed around it.' : 'Noted: off track. That area is now less likely, and it won\'t show again.';
});
function showCp(){ const r = $('cp').getBoundingClientRect(); if (r.top > window.innerHeight - 120 || r.bottom < 0) $('cp').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'}); }
function esc(s){ return String(s??'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function answer(a){
  if (!current) return; const c=current.cell, x=(c%W+.5)/W, y=(((c/W)|0)+.5)/H;
  const o = a==='love' ? {x,y,yv:1,w:2.2} : a==='closer' ? {x,y,yv:1,w:.9} : a==='no' ? {x,y,yv:0,w:2.0} : {x,y,yv:1,w:.6};
  obs.push(o); journey.obs.push(o);
  if (a==='weirder'){ kappa += .6; BETA += .4; }
  history.push({title: current.title, ans: a, setting: current.setting});
  journey.trail.push({title: current.title, ans: a, setting: current.setting, when: new Date().toISOString().slice(0,10)}); saveJourney();
  const li=document.createElement('li'); li.innerHTML=`<span>${esc(current.title)}</span><span>${a}</span>`; $('trail').appendChild(li); $('trailcard').hidden=false;
  $('sQ').textContent = history.length;
  fit(); reroutes++; $('sR').textContent = reroutes;
  const from = (a==='love'||a==='closer') ? c : anim.start;   // move forward only when the step felt right
  $('status').textContent = a==='no' ? 'Dead end. Raising the cost of that area and replanning…' : a==='weirder' ? 'Going stranger: favouring emptier, riskier ground…' : 'Noted. Moving forward and replanning…';
  $('cp').innerHTML = `<div class="tag">Checkpoint</div><h2>Replanning…</h2><p class="imgline">The model updated. Watch the fog shift and the routes regrow.</p>`;
  current=null; plan(from);
}
function arrive(s){
  anim.arrived = true; const end = s.path[s.path.length-1];
  journey.paths.push(s.path.filter((c,k)=>k%2===0 || k===s.path.length-1).map(cellXY)); journey.paths = journey.paths.slice(-12); saveJourney(); anim.cp = end; const bl = blendAt(end), setting = settingFor(bl);
  $('status').textContent = `Arrived. The fog along your route is down to ${(routeFog(s.path)*100).toFixed(0)}%.`;
  $('cp').innerHTML = `<div class="tag">Destination · no existing work lives here</div><h2 class="thinking">Rendering the work only you would make…</h2>`;
  Promise.all([words(bl, setting, history), nearest(end)]).then(([j])=>{
    $('cp').innerHTML = `<div class="tag">Destination · no existing work lives here</div>
      <div class="cpgrid">
      <div><div class="col-h">The work only you would make</div><h2>${esc(j.title)}</h2><div class="setting">${esc(j.setting||setting)}</div>
        <canvas class="sketch" id="sk"></canvas></div>
      <div><div class="col-h">Your assignment</div><p class="frag">“${esc(j.line)}”</p>
        <div class="assign"><p><b>Photo:</b> ${esc(j.image)} Find its echo where you live, in ${esc($('here').value)}. Shoot 8 frames.</p>
        <p><b>Writing:</b> Write 300 words set in ${esc(j.setting||setting)}, about ${esc($('mind').value)}.</p></div></div>
      <div><div class="col-h">Next</div><div class="answers" style="grid-template-columns:1fr"><button class="ans" id="a-again" type="button">Search again from here</button><button class="ans" id="a-reset" type="button">Start over</button></div></div>
      </div>`;
    showCp();
    paint($('sk'), bl, end*13);
    $('a-again').onclick=()=>{ anim.arrived=false; kappa+=.3; plan(end); };
    $('a-reset').onclick=start;
  });
}
function start(){
  journey.sessions = (journey.sessions||0) + 1; saveJourney();
  rnd = mulberry(7 + history.length); history=[]; reroutes=0; kappa=1; BETA=.9; anim.arrived=false;
  $('trail').innerHTML=''; $('trailcard').hidden=true; $('sQ').textContent='0'; $('sR').textContent='0';
  seedObs(); fit(); busy=true;
  $('status').textContent='Searching for uncharted places your taste points to…';
  $('cp').innerHTML = `<div class="tag">Checkpoint</div><h2>Searching…</h2><p class="imgline">Three routes are growing toward three empty spots. The Compass will stop where the best route is foggiest.</p>`;
  plan(startCell());
}
document.addEventListener('keydown', e=>{ if (e.target.tagName==='INPUT') return; const m={'1':'love','2':'closer','3':'no','4':'weirder'}[e.key]; if (m && current) answer(m); });
$('go').onclick=start;

// initial resting frame: prior fog with the example seed applied, no search yet
obs=[]; fit(); SD0 = Math.max(...SD); seedObs(); fit();
if (journey.trail.length){ journey.trail.slice(-5).forEach(t=>{ const li=document.createElement('li'); li.innerHTML=`<span>${esc(t.title)}</span><span>${esc(t.ans)} · ${esc(t.when||'')}</span>`; $('trail').appendChild(li); }); $('trailcard').hidden=false; }
if (DATA){ $('kindlegend').hidden=false; if (DATA.trained) $('status').textContent = `Seeded with ${DATA.trained} of your Another Angle picks. Press "Chart my map".`; }
if (journey.obs.length) $('status').textContent = `Welcome back. Your map remembers ${journey.obs.length} answer${journey.obs.length===1?"":"s"} and ${journey.blocks.length} block${journey.blocks.length===1?"":"s"} from ${journey.sessions} session${journey.sessions===1?'':'s'}.`;
window.addEventListener('resize', resize); if (window.ResizeObserver) new ResizeObserver(()=>{ if (Math.abs(cv.getBoundingClientRect().width-CW)>1) resize(); }).observe(cv); resize(); requestAnimationFrame(tick);
})();
