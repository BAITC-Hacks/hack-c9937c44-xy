/* Offline UI concept. All values are synthetic; no model/API calls are made. */
"use strict";
const state = {horizon:48,turbine:"both",expanded:false,rows:[]};
const $ = id => document.getElementById(id);
const mean = values => values.reduce((a,b)=>a+b,0)/values.length;
const format = (number,digits=3) => number.toLocaleString("ru-RU",{minimumFractionDigits:digits,maximumFractionDigits:digits});
const stamp = value => new Date(value).toLocaleString("ru-RU",{timeZone:"UTC",day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
function power(wind){return wind<3||wind>25 ? 0 : Math.min(1,Math.max(0,(wind**3-27)/(12**3-27)));}
function buildRows(){
  const date = $("date").value;
  if(!/^2026-02-(0[1-9]|1[0-9]|2[0-8])$/.test(date)){
    $("date").setCustomValidity("Выберите дату с 1 по 28 февраля 2026 года.");
    $("date").reportValidity();
    return false;
  }
  $("date").setCustomValidity("");
  const start=Date.parse(date+"T00:00:00Z"),day=Number(date.slice(-2));
  state.rows=Array.from({length:state.horizon},(_,i)=>{
    const wind1=8.9+2.0*Math.sin(i/6+day/5)+1.05*Math.cos(i/2.8)+.4*Math.sin(i*1.8);
    const wind2=wind1*.94+.45*Math.sin(i/4+1);
    return {time:new Date(start+i*3600000).toISOString(),wind1,wind2,power1:power(wind1),power2:power(wind2)};
  });
  return true;
}
function drawChart(){
  const w=760,h=255,left=37,right=12,top=15,bottom=30,pw=w-left-right,ph=h-top-bottom;
  const x=i=>left+i*pw/(state.rows.length-1),y=v=>top+(1-v)*ph;
  const line=key=>state.rows.map((r,i)=>`${i?"L":"M"}${x(i).toFixed(2)},${y(r[key]).toFixed(2)}`).join(" ");
  let svg=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#168979" stop-opacity=".14"/><stop offset="100%" stop-color="#168979" stop-opacity=".01"/></linearGradient></defs>`;
  for(let i=0;i<=4;i++){const v=i/4;svg+=`<line x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}" stroke="#e9eee8" stroke-dasharray="3 4"/><text x="${left-10}" y="${y(v)+3}" text-anchor="end" font-size="9" fill="#93a08f">${v.toFixed(2)}</text>`;}
  for(let i=0;i<state.horizon;i+=state.horizon===48?8:4){svg+=`<text x="${x(i)}" y="${h-8}" text-anchor="${i===0?"start":"middle"}" font-size="9" fill="#93a08f">${stamp(state.rows[i].time).replace(", "," · ")}</text>`;}
  if(state.turbine!=="2")svg+=`<path d="${line("power1")} L${x(state.horizon-1)},${y(0)} L${x(0)},${y(0)} Z" fill="url(#fill)"/><path d="${line("power1")}" fill="none" stroke="#168979" stroke-width="2.2" stroke-linejoin="round"/>`;
  if(state.turbine!=="1")svg+=`<path d="${line("power2")}" fill="none" stroke="#6086b8" stroke-width="2" stroke-linejoin="round" stroke-dasharray="5 3"/>`;
  svg+="</svg>";$("chart").innerHTML=svg;
  $("chart").setAttribute("aria-label",`Демонстрационный прогноз на ${state.horizon} ${state.horizon===24?"часа":"часов"}. ${state.turbine==="both"?"Обе турбины":"Турбина "+state.turbine}. Почасовые значения доступны в таблице ниже.`);
  const points=state.rows.map((r,i)=>`${i*72/(state.rows.length-1)},${28-r.power1*25}`).join(" ");
  $("spark").innerHTML=`<svg viewBox="0 0 72 30"><polyline points="${points}" fill="none" stroke="#258e73" stroke-width="1.4"/></svg>`;
}
function drawTable(){
  const rows=state.expanded?state.rows:state.rows.slice(0,5);
  $("rows").innerHTML=rows.map(r=>`<tr><td>${stamp(r.time)}</td><td>${format(r.power1)}</td><td>${format(r.power2)}</td><td>${format(r.wind1,1)}</td><td>${format(r.wind2,1)}</td><td>✓ В пределах</td></tr>`).join("");
  $("toggle-table").textContent=state.expanded?"Свернуть таблицу ↑":`Показать все ${state.horizon} ${state.horizon===24?"часа":"часов"} ↓`;
  $("toggle-table").setAttribute("aria-expanded",String(state.expanded));
}
function render(){
  if(!buildRows())return;
  const values=state.rows.flatMap(r=>[r.power1,r.power2]);
  $("avg-power").textContent=format(mean(values));
  $("peak-power").textContent=format(Math.max(...values));
  const peak=state.rows.find(r=>r.power1===Math.max(...values)||r.power2===Math.max(...values));
  $("peak-time").textContent=stamp(peak.time)+" UTC";
  $("avg-wind").textContent=format(mean(state.rows.flatMap(r=>[r.wind1,r.wind2])),1);
  $("valid-hours").textContent=state.horizon;
  $("t1-power").textContent=format(mean(state.rows.map(r=>r.power1)));
  $("t2-power").textContent=format(mean(state.rows.map(r=>r.power2)));
  $("range-label").textContent=`${stamp(state.rows[0].time)} — ${stamp(state.rows.at(-1).time)} · UTC`;
  drawChart();drawTable();
}
document.querySelectorAll("[data-horizon]").forEach(button=>button.addEventListener("click",()=>{
  state.horizon=Number(button.dataset.horizon);
  document.querySelectorAll("[data-horizon]").forEach(b=>{const active=b===button;b.classList.toggle("selected",active);b.setAttribute("aria-pressed",String(active));});render();
}));
document.querySelectorAll("[data-turbine]").forEach(button=>button.addEventListener("click",()=>{
  state.turbine=button.dataset.turbine;
  document.querySelectorAll("[data-turbine]").forEach(b=>{const active=b===button;b.classList.toggle("selected",active);b.setAttribute("aria-pressed",String(active));});drawChart();
}));
$("date").addEventListener("change",render);
$("calculate").addEventListener("click",()=>{render();$("agent-status").textContent="Демо обновлено. GPU и API не запускались.";});
$("toggle-table").addEventListener("click",()=>{state.expanded=!state.expanded;drawTable();});
$("download").addEventListener("click",()=>{
  if(!buildRows())return;
  const origin=state.rows[0].time;
  const lines=["mode,forecast_origin,valid_time,turbine_id,power_normalized,wind_speed_ms"];
  state.rows.forEach(r=>[1,2].forEach(id=>lines.push(`synthetic-demo,${origin},${r.time},${id},${r["power"+id].toFixed(6)},${r["wind"+id].toFixed(3)}`)));
  const url=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/csv;charset=utf-8"}));
  const a=document.createElement("a");a.href=url;a.download=`DEMO_forecast_${$("date").value}_${state.horizon}h.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
