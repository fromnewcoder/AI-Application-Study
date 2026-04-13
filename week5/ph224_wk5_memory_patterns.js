function showPage(id){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}

let recentTurns=[];
let rollingSum="";
let totalTurns=0;
const THRESHOLD=4;
const KEEP=2;

function updateMeta(){
  document.getElementById('turnCount').textContent=totalTurns;
  document.getElementById('recentCount').textContent=recentTurns.length;
  document.getElementById('hasSummary').textContent=rollingSum?'yes':'none';
  if(rollingSum){
    document.getElementById('summarySection').style.display='block';
    document.getElementById('summaryText').textContent=rollingSum;
  }
}

function addMsg(role,text,cls){
  const area=document.getElementById('chatArea');
  const d=document.createElement('div');
  d.className='msg msg-'+(cls||role);
  d.innerHTML=`<span class="msg-role">${role==='user'?'You':role==='assistant'?'Claude':role}</span><span class="msg-text">${text}</span>`;
  area.appendChild(d);
  area.scrollTop=area.scrollHeight;
  return d;
}

function resetChat(){
  recentTurns=[];rollingSum="";totalTurns=0;
  document.getElementById('chatArea').innerHTML='';
  document.getElementById('summarySection').style.display='none';
  updateMeta();
}

async function callClaude(system,messages){
  const r=await fetch("https://api.minimaxi.com/anthropic/v1/messages",{
    method:"POST",
    headers:{
      "Content-Type":"application/json",
      "x-api-key":"xxx"

    },
    body:JSON.stringify({model:"MiniMax-M2.7",max_tokens:1000,system,messages})
  });
  const d=await r.json();
  return d.content?.[0]?.text||"(no response)";
}

async function summarise(){
  const toCompress=recentTurns.slice(0,-KEEP);
  recentTurns=recentTurns.slice(-KEEP);
  const prevCtx=rollingSum?`Previous summary:\n${rollingSum}\n\n`:"";
  const prompt=prevCtx+"New conversation segment:\n"+
    toCompress.map(m=>`${m.role}: ${m.content}`).join("\n")+
    "\n\nWrite a concise summary that preserves key facts, decisions, names, and preferences mentioned.";
  const thinking=addMsg('system','Summarising older turns…','system');
  rollingSum=await callClaude("You are a concise summariser. Return only the summary, no preamble.",[{role:"user",content:prompt}]);
  thinking.remove();
  addMsg('system','Summary updated — older turns compressed','system');
}

async function sendMsg(){
  const inp=document.getElementById('userInput');
  const text=inp.value.trim();
  if(!text)return;
  inp.value='';
  addMsg('user',text);
  recentTurns.push({role:"user",content:text});
  totalTurns++;
  updateMeta();

  const system=rollingSum
    ?"You are a helpful assistant.\n\nEarlier conversation (summary):\n"+rollingSum
    :"You are a helpful assistant.";

  const thinking=addMsg('assistant','…','assistant');
  thinking.querySelector('.msg-text').classList.add('thinking');

  let reply;
  try{
    reply=await callClaude(system,recentTurns);
  }catch(e){
    reply="API error: "+e.message;
  }
  thinking.querySelector('.msg-text').classList.remove('thinking');
  thinking.querySelector('.msg-text').textContent=reply;
  recentTurns.push({role:"assistant",content:reply});
  totalTurns++;
  updateMeta();

  if(recentTurns.length>=THRESHOLD){
    await summarise();
    updateMeta();
  }
}