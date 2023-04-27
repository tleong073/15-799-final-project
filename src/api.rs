/// The `api` module is the public entry point for all FrodoPIR database.
use crate::db::Database;
pub use crate::db::{BaseParams, CommonParams};
use crate::errors::{
  ErrorOverflownAdd, ErrorQueryParamsReused, ResultBoxedError,
};
pub use crate::utils::format::*;
use crate::utils::lwe::*;
use crate::utils::matrices::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::str;
use rayon::prelude::*;

/// A `Shard` is an instance of a database, where each row corresponds
/// to a single element, that has been preprocessed by the server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
  db: Database,
  base_params: BaseParams,
}
impl Shard {
  /// Expects a JSON file of base64-encoded strings in file path. It also
  /// expects the lwe dimension, m (the number of DB elements), element size
  /// (in bytes) of the database elements, and plaintext bits.
  /// It will call the 'from_base64_strings' function to generate the database.
  pub fn from_json_file(
    file_path: &str,
    lwe_dim: usize,
    m: usize,
    ele_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let file_contents: String =
      fs::read_to_string(file_path).unwrap().parse().unwrap();
    let elements: Vec<String> = serde_json::from_str(&file_contents).unwrap();
    Shard::from_base64_strings(&elements, lwe_dim, m, ele_size, plaintext_bits)
  }

  /// Expects an array of base64-encoded strings and converts into a
  /// database that can process client queries
  pub fn from_base64_strings(
    base64_strs: &[String],
    lwe_dim: usize,
    m: usize,
    ele_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let db = Database::new(base64_strs, m, ele_size, plaintext_bits)?;
    let base_params = BaseParams::new(&db, lwe_dim);
    Ok(Self { db, base_params })
  }

  /// Write base_params and DB to file
  pub fn write_to_file(
    &self,
    db_path: &str,
    params_path: &str,
  ) -> ResultBoxedError<()> {
    self.db.write_to_file(db_path)?;
    self.base_params.write_to_file(params_path)?;
    Ok(())
  }

  // Produces a serialized response (base64-encoded) to a serialized
  // client query
  pub fn respond(&self, q: &Query) -> ResultBoxedError<Vec<u8>> {
    let resp = Response(
      (0..self.db.get_matrix_width_self())
        .into_iter()
        .map(|i| self.db.vec_mult(q.as_slice(), i))
        .collect(),
    );
    let se = bincode::serialize(&resp);

    Ok(se?)
  }

  // Produces a serialized response (base64-encoded) to a serialized
  // client query
  pub fn respond_par(&self, q: &Query) -> ResultBoxedError<Vec<u8>> {
    let resp = Response(
      (0..self.db.get_matrix_width_self())
        .into_par_iter()
        .map(|i| self.db.vec_mult_par(q.as_slice(), i))
        .collect(),
    );
    let se = bincode::serialize(&resp);

    Ok(se?)
  }


   /// Returns the database
   pub fn get_db(&self) -> &Database {
    &self.db
  }

  /// Returns the base parameters
  pub fn get_base_params(&self) -> &BaseParams {
    &self.base_params
  }

  pub fn into_row_iter(&self) -> std::vec::IntoIter<std::string::String> {
    (0..self.get_db().get_matrix_height())
      .into_iter()
      .map(|i| self.get_db().get_db_entry(i))
      .collect::<Vec<String>>()
      .into_iter()
  }
}

  /// Struct for representing database as multiple shards 
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardedDB {
  pub sharded_db: Vec<Shard>,
  pub num_shards: u32,
}
impl ShardedDB {
  /// Expects a JSON file of vectors of base64-encoded strings in file path. It also
  /// expects the lwe dimension, m (the number of DB elements), element size
  /// (in bytes) of the database elements, and plaintext bits.
  /// It will call the 'from_base64_strings' function to generate the database.
  pub fn from_json_file(
    file_path: &str,
    lwe_dim: usize,
    m: usize,
    ele_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {

    // Read in JSON to string 
    let file_contents: String =
      fs::read_to_string(file_path).unwrap().parse().unwrap();

    let shards: Vec<Vec<String>> = serde_json::from_str(&file_contents).unwrap();
    
    Ok(
      Self{
        sharded_db: shards.iter()
            .map(|s| Shard::from_base64_strings(&s, lwe_dim, m, ele_size, plaintext_bits).unwrap())
            .collect(),
        num_shards: shards.len() as u32,
      }
    )
  }

  // Produces a vector of serialized responses (base64-encoded) to a
  // serialized client query
  pub fn respond(&self, q: &Query) -> ResultBoxedError<Vec<Vec<u8>>> {

    let resps = self.sharded_db
        .par_iter()
        .map(|s| s.respond_par(q).unwrap())
        .collect();

    Ok(resps)
  }

  /// Returns the database
  pub fn get_sharded_db(&self) -> &Vec<Shard> {
    &self.sharded_db
  }

  /// Returns the base parameters
  pub fn get_length(&self) -> u32 {
    self.num_shards
  }
}

/// The `QueryParams` struct is initialized to be used for a client
/// query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryParams {
  lhs: Vec<u32>,
  rhs: Vec<u32>,
  ele_size: usize,
  plaintext_bits: usize,
  pub used: bool,
}
impl QueryParams {
  pub fn new(cp: &CommonParams, bp: &BaseParams) -> ResultBoxedError<Self> {
    let s = random_ternary_vector(bp.get_dim());
    Ok(Self {
      lhs: cp.mult_left(&s)?,
      rhs: bp.mult_right(&s)?,
      ele_size: bp.get_ele_size(),
      plaintext_bits: bp.get_plaintext_bits(),
      used: false,
    })
  }

  /// Prepares a new client query based on an input row_index
  /// Computes b~ from b and row index
  pub fn prepare_query(&mut self, row_index: usize) -> ResultBoxedError<Query> {
    if self.used {
      return Err(Box::new(ErrorQueryParamsReused {}));
    }
    self.used = true;
    let query_indicator = get_rounding_factor(self.plaintext_bits);
    let mut lhs = Vec::new();
    lhs.clone_from(&self.lhs.clone());
    // Add q/p to desired index
    let (result, check) = lhs[row_index].overflowing_add(query_indicator);
    if !check {
      lhs[row_index] = result;
    } else {
      return Err(Box::new(ErrorOverflownAdd {}));
    }
    Ok(Query(lhs))
  }
}

/// The `Query` struct holds the necessary information encoded in
/// a client PIR query to the server DB for a particular `row_index`. It
/// provides methods for parsing server responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query(Vec<u32>);
impl Query {
  pub fn as_slice(&self) -> &[u32] {
    &self.0
  }
}

/// The `Response` object wraps a response from a single shard
#[derive(Clone, Serialize, Deserialize)]
pub struct Response(Vec<u32>);
impl Response {
  pub fn as_slice(&self) -> &[u32] {
    &self.0
  }

  /// Parses the output as a row of u32 values
  pub fn parse_output_as_row(&self, qp: &QueryParams) -> Vec<u32> {
    // get parameters for rounding
    let rounding_factor = get_rounding_factor(qp.plaintext_bits);
    let rounding_floor = get_rounding_floor(qp.plaintext_bits);
    let plaintext_size = get_plaintext_size(qp.plaintext_bits);

    // perform division and rounding
    (0..Database::get_matrix_width(qp.ele_size, qp.plaintext_bits))
      .into_iter()
      .map(|i| {
        let unscaled_res = self.0[i].wrapping_sub(qp.rhs[i]);
        let scaled_res = unscaled_res / rounding_factor;
        let scaled_rem = unscaled_res % rounding_factor;
        let mut rounded_res = scaled_res;
        if scaled_rem > rounding_floor {
          rounded_res += 1;
        }
        rounded_res % plaintext_size
      })
      .collect()
  }

  /// Parses the output as bytes
  pub fn parse_output_as_bytes(&self, qp: &QueryParams) -> Vec<u8> {
    let row = self.parse_output_as_row(qp);
    bytes_from_u32_slice(&row, qp.plaintext_bits, qp.ele_size)
  }

  /// Parses the output as a base64-encoded string
  pub fn parse_output_as_base64(&self, qp: &QueryParams) -> String {
    let row = self.parse_output_as_row(qp);
    base64_from_u32_slice(&row, qp.plaintext_bits, qp.ele_size)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand_core::{OsRng, RngCore};

  //#[test]
  fn client_query_to_server_10_times() {
    let m = 2u32.pow(16) as usize;
    let ele_size = 2u32.pow(8) as usize;
    let plaintext_bits = 12usize;
    let lwe_dim = 512;
    let db_eles = generate_db_eles(m, (ele_size + 7) / 8);
    let shard = Shard::from_base64_strings(
      &db_eles,
      lwe_dim,
      m,
      ele_size,
      plaintext_bits,
    )
    .unwrap();
    let bp = shard.get_base_params();
    let cp = CommonParams::from(bp);
    #[allow(clippy::needless_range_loop)]
    for i in 0..10 {
      let mut qp = QueryParams::new(&cp, bp).unwrap();
      let q = qp.prepare_query(i).unwrap();
      let d_resp = shard.respond(&q).unwrap();
      let resp: Response = bincode::deserialize(&d_resp).unwrap();
      let output = resp.parse_output_as_base64(&qp);
      assert_eq!(output, db_eles[i]);
    }
  }

  #[test]
  fn client_query_to_server_10_times_sharded_sequential() {
    let ele_size = 2u32.pow(8) as usize;
    let plaintext_bits = 12usize;
    let lwe_dim = 512;
    let num_shards = 16;
    let m = (2u32.pow(16)/num_shards) as usize;

    let shard_vec = vec![0; num_shards as usize];
    let tmp_dbs = vec![""; num_shards as usize];

    let db_vec:Vec<Vec<String>> = tmp_dbs.par_iter()
        .map(|x| generate_db_eles(m, (ele_size + 7) / 8))
        .collect();
    // Generate Shards
    let shards: Vec<Shard> = db_vec.par_iter()
        .map(|db_eles| Shard::from_base64_strings(
          &db_eles,
          lwe_dim,
          m,
          ele_size,
          plaintext_bits,
        ).unwrap())
        .collect();

    let sharded_db = ShardedDB{
      sharded_db: shards,
      num_shards: num_shards as u32,
    };
    let bp = sharded_db.sharded_db[0].get_base_params();
    let cp = CommonParams::from(bp);
    #[allow(clippy::needless_range_loop)]
    for i in 0..10 {
      let mut qp = QueryParams::new(&cp, bp).unwrap();
      let q = qp.prepare_query(i).unwrap();
      let d_resps: Vec<Vec<u8>> = sharded_db.respond(&q).unwrap();
      //let resps: Vec<Response> = bincode::deserialize(&d_resp).unwrap();

      let outputs: Vec<String> = d_resps.par_iter()
        .map(|d| bincode::deserialize(&d).unwrap())
        .map(|r: Response| r.parse_output_as_base64(&qp))
        .collect();

      //let output = resp.parse_output_as_base64(&qp);
      //let output2 = resp.parse_output_as_base64(&qp);

      assert_eq!(outputs[0], db_vec[0][i]);
      //assert_eq!(outputs[1], db_vec[i]);
    }
  }

  #[test]
  fn client_query_to_server_attempt_params_reuse() {
    let m = 2u32.pow(6) as usize;
    let ele_size = 2u32.pow(8) as usize;
    let plaintext_bits = 10usize;
    let lwe_dim = 512;
    let db_eles = generate_db_eles(m, (ele_size + 7) / 8);
    let shard = Shard::from_base64_strings(
      &db_eles,
      lwe_dim,
      m,
      ele_size,
      plaintext_bits,
    )
    .unwrap();
    let bp = shard.get_base_params();
    let cp = CommonParams::from(bp);
    let mut qp = QueryParams::new(&cp, bp).unwrap();
    // should be successful in generating a query
    let res_unused = qp.prepare_query(0);
    assert!(res_unused.is_ok());

    // should be "used"
    assert!(qp.used);

    // should be successful in generating a query
    let res = qp.prepare_query(0);
    assert!(res.is_err());
  }

  fn generate_db_eles(num_eles: usize, ele_byte_len: usize) -> Vec<String> {
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
