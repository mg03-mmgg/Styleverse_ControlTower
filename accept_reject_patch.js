/* ---------- applying a decision closes the loop ---------- */

// Writes the decision to the database so it persists beyond the browser
// session. Fire-and-forget: the UI updates instantly either way, so a
// slow or failed API call never blocks the planner's workflow — the
// failure is logged to the console instead.
function _recordDecision(r, c, decisionType){
  var payload = {
    product_id: c.sku.id,
    location_id: c.store.id,
    recommendation_type: r.t,
    decision_type: decisionType,
    recommended_quantity: r.qty ? Math.round(r.qty) : null,
    recommended_price: (r.t === 'Markdown' && r.depth != null)
        ? Math.round(c.sku.price * (1 - r.depth / 100) * 100) / 100
        : null,
    expected_margin: r.val != null ? Math.round(r.val * 100) / 100 : null,
    reason: r.why || null,
    planner_id: "demo_planner"
  };

  fetch(API_BASE + "/api/dashboard-decision", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  })
  .then(function(res){ return res.json(); })
  .then(function(data){
    console.log("[decision] saved:", data.decision_id, data.status,
                r.t, c.sku.id, c.store.id);
  })
  .catch(function(err){
    console.error("[decision] could not save to database:", err);
  });
}

function accept(idx){
  var r=QUEUE[idx],c=r.c;
  if(r.t==='Replenish'||r.t==='Re-cut'){c.inTransit+=r.qty;c.recv+=r.qty}
  else if(r.t==='Transfer'){r.from.onHand-=r.qty;c.onHand+=r.qty}
  else if(r.t==='Markdown'){c.md=r.depth}
  c.accepted=r.t;
  RECOVERED+=Math.max(0,r.val);
  LOG.unshift({t:r.t,sku:c.sku.id,store:c.store.id,val:r.val,d:'Accepted'});
  _recordDecision(r, c, 'Accept');
  run();render();
}

function reject(idx){
  var r=QUEUE[idx];r.c.accepted='Rejected';
  LOG.unshift({t:r.t,sku:r.c.sku.id,store:r.c.store.id,val:0,d:'Rejected'});
  _recordDecision(r, r.c, 'Reject');
  run();render();
}