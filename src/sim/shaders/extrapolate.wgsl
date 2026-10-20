// Two layers of velocity extrapolation into air and solid faces (on each component's own
// lattice), in one pass and in place: a face without a valid velocity takes the average of its
// valid neighbours', or, with none, the average of what those neighbours take so. (The same as
// two passes of one layer, ping-ponging buffers, without the second set: the faces it writes are
// never the ones it reads, the valid, which keep theirs.)

@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read_write> vel: array<f32>;

fn neighbour(p: vec3i, d: u32) -> vec3i {
  var o = vec3i(0);
  o[d / 2u] = select(-1, 1, (d & 1u) == 1u);
  return p + o;
}

@compute @workgroup_size(8, 8, 4)
fn extrapolate(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.s;
  if (gid.x >= s || gid.y >= s || gid.z >= 3u * s) { return; }
  let comp = gid.z / s;
  let k = gid.z % s;
  if (farOutside(facePos(comp, gid.x, gid.y, k))) { return; }
  let base = comp * P.faceCount;
  let t = base + gid.x + s * (gid.y + s * k);
  if (valid[t] != 0u) { return; }
  let p = vec3i(i32(gid.x), i32(gid.y), i32(k));
  let si = i32(s);
  // The first layer: the valid neighbours' average.
  var sum = 0.0;
  var cnt = 0.0;
  for (var d = 0u; d < 6u; d++) {
    let q = neighbour(p, d);
    if (all(q >= vec3i(0)) && all(q < vec3i(si))) {
      let tq = base + faceIdx(q.x, q.y, q.z);
      if (valid[tq] != 0u) {
        sum += vel[tq];
        cnt += 1.0;
      }
    }
  }
  if (cnt > 0.0) {
    vel[t] = sum / cnt;
    return;
  }
  // The second: the average of the neighbours' first layers, where they took one (none of them
  // is valid, or this face would have taken theirs; and those far outside the glass take none).
  for (var d = 0u; d < 6u; d++) {
    let q = neighbour(p, d);
    if (all(q >= vec3i(0)) && all(q < vec3i(si)) && !farOutside(facePos(comp, u32(q.x), u32(q.y), u32(q.z)))) {
      var qSum = 0.0;
      var qCnt = 0.0;
      for (var e = 0u; e < 6u; e++) {
        let r = neighbour(q, e);
        if (all(r >= vec3i(0)) && all(r < vec3i(si))) {
          let tr = base + faceIdx(r.x, r.y, r.z);
          if (valid[tr] != 0u) {
            qSum += vel[tr];
            qCnt += 1.0;
          }
        }
      }
      if (qCnt > 0.0) {
        sum += qSum / qCnt;
        cnt += 1.0;
      }
    }
  }
  if (cnt > 0.0) { vel[t] = sum / cnt; }
}
