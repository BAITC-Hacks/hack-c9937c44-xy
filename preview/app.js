"use strict";
const state = {horizon:48,turbine:"both",expanded:false,rows:[],csv:"",request:0};
const $ = id => document.getElementById(id);
const mean = values => values.reduce((a,b)=>a+b,0)/values.length;
const format = (number,digits=3) => number.toLocaleString("ru-RU",{minimumFractionDigits:digits,maximumFractionDigits:digits});
const stamp = value => new Date(value).toLocaleString("ru-RU",{timeZone:"UTC",day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
async function loadRows(){
  const date=$("date");
  if(!date.value || !date.checkValidity())throw new Error("Выберите дату с января по февраль 2026 года.");
  const source=$("source").value;
  const stem=`forecast_${date.value.replaceAll("-","")}T0000Z`;
  const base=new URL(`../outputs/${source}/${stem}`,location.href);
  const [csvResponse,auditResponse]=await Promise.all([fetch(base+".csv"),fetch(base+".json")]);
  if(!csvResponse.ok || !auditResponse.ok)throw new Error(`Файл ${stem} не найден в outputs/${source}. Сначала запустите Python-расчёт.`);
  const [csv,audit]=await Promise.all([csvResponse.text(),auditResponse.json()]);
  const expectedMode=source==="demo"?"synthetic-demo":"historical-backtest";
  const origin=Date.parse(date.value+"T00:00:00Z");
  if(audit.mode!==expectedMode || Date.parse(audit.forecast_origin)!==origin ||
     (source==="backtest" && audit.trained_model!==true))throw new Error("CSV-аудит не соответствует выбранному источнику и дате.");
  const lines=csv.trim().split(/\r?\n/);
  if(lines.shift()!=="forecast_origin,valid_time,turbine_id,power_normalized,wind_speed_ms" ||
     lines.length!==audit.horizon_hours*2 || audit.horizon_hours<state.horizon)throw new Error("Неполный CSV или неверный горизонт прогноза.");
  const hours=new Map();
  for(const line of lines){
    const fields=line.split(",");
    if(fields.length!==5 || Date.parse(fields[0])!==origin || !["turbine_1","turbine_2"].includes(fields[2]))throw new Error("Некорректная строка прогноза.");
    const time=Date.parse(fields[1]),power=Number(fields[3]),wind=Number(fields[4]);
    if(!Number.isFinite(time) || !Number.isFinite(power) || !Number.isFinite(wind) || power<0 || power>1 || wind<0)throw new Error("Некорректные значения прогноза.");
    const hour=(time-origin)/3600000;
    if(!Number.isInteger(hour) || hour<0 || hour>=audit.horizon_hours)throw new Error("Время прогноза вне горизонта.");
    const row=hours.get(hour)||{time:new Date(time).toISOString()};
    const id=fields[2].slice(-1);
    if(row["power"+id]!==undefined)throw new Error("Повтор турбины в CSV.");
    row["power"+id]=power;row["wind"+id]=wind;hours.set(hour,row);
  }
  const rows=Array.from({length:state.horizon},(_,hour)=>hours.get(hour));
  if(rows.some(row=>!row || row.power1===undefined || row.power2===undefined))throw new Error("В CSV пропущены часы или турбины.");
  return {rows,csv,audit,stem,source};
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
  $("chart").setAttribute("aria-label",`Прогноз на ${state.horizon} ${state.horizon===24?"часа":"часов"}. ${state.turbine==="both"?"Обе турбины":"Турбина "+state.turbine}. Почасовые значения доступны в таблице ниже.`);
  const points=state.rows.map((r,i)=>`${i*72/(state.rows.length-1)},${28-r.power1*25}`).join(" ");
  $("spark").innerHTML=`<svg viewBox="0 0 72 30"><polyline points="${points}" fill="none" stroke="#258e73" stroke-width="1.4"/></svg>`;
}
function drawTable(){
  const rows=state.expanded?state.rows:state.rows.slice(0,5);
  $("rows").innerHTML=rows.map(r=>`<tr><td>${stamp(r.time)}</td><td>${format(r.power1)}</td><td>${format(r.power2)}</td><td>${format(r.wind1,1)}</td><td>${format(r.wind2,1)}</td><td>✓ В пределах</td></tr>`).join("");
  $("toggle-table").textContent=state.expanded?"Свернуть таблицу ↑":`Показать все ${state.horizon} ${state.horizon===24?"часа":"часов"} ↓`;
  $("toggle-table").setAttribute("aria-expanded",String(state.expanded));
}
async function render(){
  const request=++state.request;
  $("agent-status").textContent="Загрузка прогноза…";
  try{
    const {rows,csv,audit,stem,source}=await loadRows();
    if(request!==state.request)return;
    state.rows=rows;state.csv=csv;state.stem=stem;
    $("download").disabled=false;
    const demo=source==="demo";
    $("source-note").textContent=demo?
      "Синтетический офлайн-прогноз из Python. Это не результат модели и не оценка точности.":
      `Прогноз модели из CSV. История: ${audit.history_policy||"не указана"}. Качество за февраль не оценено.`;
    $("table-description").textContent=(demo?"Синтетический расчёт Python":"Выход модели")+" · UTC";
    $("wind-detail").textContent=demo?"Иллюстративный погодный сценарий":"Архивный прогноз ветра на 10 м";
    $("mode-badge").textContent=demo?"ДЕМО":"МОДЕЛЬ";
    $("physics-rule").innerHTML="Диапазон мощности 0–1<br>"+
      ((audit.physical_validation?.wind_limits_applied ?? demo)?"Пороги ветра 3 / 25 м/с":"Ветер 10 м: пороги не применяются");
    document.querySelectorAll(".demo-chip").forEach(chip=>chip.textContent=demo?"Демо":"Модель");
    const values=rows.flatMap(r=>[r.power1,r.power2]);
    $("avg-power").textContent=format(mean(values));
    $("peak-power").textContent=format(Math.max(...values));
    const peak=rows.find(r=>r.power1===Math.max(...values)||r.power2===Math.max(...values));
    $("peak-time").textContent=stamp(peak.time)+" UTC";
    $("avg-wind").textContent=format(mean(rows.flatMap(r=>[r.wind1,r.wind2])),1);
    $("valid-hours").textContent=state.horizon;
    $("t1-power").textContent=format(mean(rows.map(r=>r.power1)));
    $("t2-power").textContent=format(mean(rows.map(r=>r.power2)));
    $("range-label").textContent=`${stamp(rows[0].time)} — ${stamp(rows.at(-1).time)} · UTC`;
    $("agent-status").textContent=demo?"Загружен офлайн-расчёт Python":"Загружен результат модели";
    drawChart();drawTable();
  }catch(error){
    if(request!==state.request)return;
    state.rows=[];state.csv="";$("download").disabled=true;
    $("source-note").textContent=error.message;
    $("agent-status").textContent="Прогноз не загружен";
    $("table-description").textContent="Нет данных · UTC";
    $("wind-detail").textContent="—";
    $("mode-badge").textContent="НЕТ ДАННЫХ";
    $("physics-rule").textContent="Нет данных";
    document.querySelectorAll(".demo-chip").forEach(chip=>chip.textContent="—");
    $("chart").textContent="Нет данных для выбранной даты";
    $("chart").setAttribute("aria-label","Нет данных для выбранной даты");
    $("rows").textContent="";$("spark").textContent="";
    $("toggle-table").textContent="Нет данных";
    for(const id of ["avg-power","peak-power","peak-time","avg-wind","valid-hours","t1-power","t2-power","range-label"])$(id).textContent="—";
  }
}
document.querySelectorAll("[data-horizon]").forEach(button=>button.addEventListener("click",()=>{
  state.horizon=Number(button.dataset.horizon);
  document.querySelectorAll("[data-horizon]").forEach(b=>{const active=b===button;b.classList.toggle("selected",active);b.setAttribute("aria-pressed",String(active));});render();
}));
document.querySelectorAll("[data-turbine]").forEach(button=>button.addEventListener("click",()=>{
  state.turbine=button.dataset.turbine;
  document.querySelectorAll("[data-turbine]").forEach(b=>{const active=b===button;b.classList.toggle("selected",active);b.setAttribute("aria-pressed",String(active));});if(state.rows.length)drawChart();
}));
$("source").addEventListener("change",render);
$("date").addEventListener("change",render);
$("calculate").addEventListener("click",render);
$("toggle-table").addEventListener("click",()=>{if(!state.rows.length)return;state.expanded=!state.expanded;drawTable();});
$("download").addEventListener("click",()=>{
  if(!state.csv)return;
  const url=URL.createObjectURL(new Blob([state.csv],{type:"text/csv;charset=utf-8"}));
  const a=document.createElement("a");a.href=url;a.download=state.stem+".csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
