use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use frodo_pir::api::{CommonParams, QueryParams, Response, Shard, ShardedDB};
use pi_rs_cli_utils::*;
use std::time::Duration;
use rayon::prelude::*;

const BENCH_ONLINE: bool = true;
const BENCH_DB_GEN: bool = false;

fn criterion_benchmark(c: &mut Criterion) {
  let CLIFlags {
    m,
    lwe_dim,
    ele_size,
    plaintext_bits,
    num_shards,
    ..
  } = parse_from_env();
  let mut lwe_group = c.benchmark_group("lwe");

  println!("Setting up DB for benchmarking. This might take a while...");
  let db_eles = bench_utils::generate_db_eles(m, (ele_size + 7) / 8);
  let shard =
    Shard::from_base64_strings(&db_eles, lwe_dim, m, ele_size, plaintext_bits)
      .unwrap();
  //Sharded DB setup
  let shard_vec = vec![0; num_shards as usize];
  let tmp_dbs = vec![""; num_shards as usize];

  let db_vec:Vec<Vec<String>> = tmp_dbs.par_iter()
        .map(|x| bench_utils::generate_db_eles(m/num_shards, (ele_size + 7) / 8))
        .collect();
  
  // Generate Shards
  let shards: Vec<Shard> = db_vec.par_iter()
        .map(|db_eles| Shard::from_base64_strings(
          &db_eles,
          lwe_dim,
          m/num_shards,
          ele_size,
          plaintext_bits,
        ).unwrap())
        .collect();
  
  let sharded_db = ShardedDB{
    sharded_db: shards,
    num_shards: num_shards as u32,
  };

  println!("Setup complete, starting benchmarks");
  if BENCH_ONLINE {
    _bench_client_query(&mut lwe_group, &shard);
    _bench_client_query_par(&mut lwe_group, &sharded_db);
  }
  if BENCH_DB_GEN {
    lwe_group.sample_size(10);
    lwe_group.measurement_time(Duration::from_secs(100)); // To remove a warning, you can increase this to 500 or more.
    _bench_db_generation(&mut lwe_group, &shard, &db_eles);
  }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn _bench_db_generation(
  c: &mut BenchmarkGroup<criterion::measurement::WallTime>,
  shard: &Shard,
  db_eles: &[String],
) {
  let db = shard.get_db();
  let bp = shard.get_base_params();
  let w = db.get_matrix_width_self();

  c.bench_function(
    format!(
      "derive LHS from seed, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| CommonParams::from(bp));
    },
  );

  println!("Starting DB generation benchmarks");
  c.bench_function(
    format!(
      "generate db and params, m: {}, w: {}",
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        Shard::from_base64_strings(
          db_eles,
          bp.get_dim(),
          db.get_matrix_height(),
          db.get_ele_size(),
          db.get_plaintext_bits(),
        )
        .unwrap();
      });
    },
  );
  println!("Finished DB generation benchmarks");
}

fn _bench_client_query(
  c: &mut BenchmarkGroup<criterion::measurement::WallTime>,
  shard: &Shard,
) {
  let db = shard.get_db();
  let bp = shard.get_base_params();
  let cp = CommonParams::from(bp);
  let w = db.get_matrix_width_self();
  let idx = 10;

  println!("Starting client query benchmarks");
  let mut _qp = QueryParams::new(&cp, bp).unwrap();
  let _q = _qp.prepare_query(idx).unwrap();
  let mut _resp = shard.respond(&_q).unwrap();
  c.bench_function(
    format!(
      "create client query params, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| QueryParams::new(&cp, bp));
    },
  );

  c.bench_function(
    format!(
      "client query prepare, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        _qp.used = false;
        _qp.prepare_query(idx).unwrap();
      });
    },
  );

  c.bench_function(
    format!(
      "server response compute, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        shard.respond(&_q).unwrap();
      });
    },
  );

  c.bench_function(
    format!(
      "client parse server response, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        let deser: Response = bincode::deserialize(&_resp).unwrap();
        deser.parse_output_as_base64(&_qp);
      });
    },
  );
  println!("Finished client query benchmarks");
}

fn _bench_client_query_par(
  c: &mut BenchmarkGroup<criterion::measurement::WallTime>,
  sharded_db: &ShardedDB,
) {
  let shards = sharded_db.get_sharded_db();
  let bp = shards[0].get_base_params();
  let cp = CommonParams::from(bp);
  let db = shards[0].get_db();
  let w = db.get_matrix_width_self();
  let idx = 10;

  println!("Starting client query benchmarks");
  let mut _qp = QueryParams::new(&cp, bp).unwrap();
  let _q = _qp.prepare_query(idx).unwrap();
  let mut _resp = shards[0].respond(&_q).unwrap();
  c.bench_function(
    format!(
      "PAR create client query params, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| QueryParams::new(&cp, bp));
    },
  );

  c.bench_function(
    format!(
      "PAR client query prepare, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        _qp.used = false;
        _qp.prepare_query(idx).unwrap();
      });
    },
  );

  c.bench_function(
    format!(
      "PAR server response compute, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        sharded_db.respond(&_q).unwrap();
      });
    },
  );

  c.bench_function(
    format!(
      "PAR client parse server response, lwe_dim: {}, m: {}, w: {}",
      bp.get_dim(),
      db.get_matrix_height(),
      w
    ),
    |b| {
      b.iter(|| {
        let deser: Response = bincode::deserialize(&_resp).unwrap();
        deser.parse_output_as_base64(&_qp);
      });
    },
  );
  println!("Finished PAR client query benchmarks");
}

mod bench_utils {
  use rand_core::{OsRng, RngCore};
  pub fn generate_db_eles(num_eles: usize, ele_byte_len: usize) -> Vec<String> {
    let mut eles = Vec::with_capacity(num_eles);
    for _ in 0..num_eles {
      let mut ele = vec![0u8; ele_byte_len];
      OsRng.fill_bytes(&mut ele);
      let ele_str = base64::encode(ele);
      eles.push(ele_str);
    }
    eles
  }
}
