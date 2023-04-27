use clap::{App, Arg};
use std::env;

pub struct CLIFlags {
  pub m: usize,
  pub lwe_dim: usize,
  pub plaintext_bits: usize,
  pub ele_size: usize,
  pub num_shards: usize,
}

pub fn parse_cli_flags() -> CLIFlags {
  let matches = App::new("PIR example")
    .version("0.0.1")
    .author("Alex Davidson <coela@alxdavids.xyz>")
    .about("Flags for setting PIR parameters")
    .arg(
      Arg::with_name("matrix_height")
        .short("m")
        .long("matrix_height")
        .takes_value(true)
        .default_value("16")
        .help("Log2 of height of DB matrix"),
    )
    .arg(
      Arg::with_name("ele_size")
        .short("e")
        .long("ele_size")
        .takes_value(true)
        .default_value("13")
        .help("Log2 of element bit length"),
    )
    .arg(
      Arg::with_name("plaintext_bits")
        .short("p")
        .long("plaintext_bits")
        .takes_value(true)
        .default_value("10")
        .help("Number of plaintext bits encoded in each entry of DB matrix"),
    )
    .arg(
      Arg::with_name("dim")
        .short("d")
        .long("dim")
        .takes_value(true)
        .default_value("2048")
        .help("LWE dimension"),
    )
    .arg(
      Arg::with_name("num_shards")
        .short("s")
        .long("num_shards")
        .takes_value(true)
        .default_value("8")
        .help("NUmber of shards in database"),
    )
    .get_matches();

  let ele_size =
    parse_exp_to_usize(String::from(matches.value_of("ele_size").unwrap()));
  let lwe_dim: usize = String::from(matches.value_of("dim").unwrap())
    .parse()
    .unwrap();
  let plaintext_bits: usize =
    String::from(matches.value_of("plaintext_bits").unwrap())
      .parse()
      .unwrap();
  let m = parse_exp_to_usize(String::from(
    matches.value_of("matrix_height").unwrap(),
  ));
  let num_shards = parse_exp_to_usize(String::from(
    matches.value_of("num_shards").unwrap(),
  ));
  CLIFlags {
    m,
    lwe_dim,
    plaintext_bits,
    ele_size,
    num_shards
  }
}

pub fn parse_from_env() -> CLIFlags {
  let ele_size = parse_exp_to_usize(env::var("PIR_ELE_SIZE_EXP").unwrap());
  let lwe_dim: usize = env::var("PIR_LWE_DIM").unwrap().parse().unwrap();
  // Parse num shards
  let num_shards = parse_exp_to_usize(env::var("PIR_NUM_SHARDS").unwrap());
  let plaintext_bits: usize =
    env::var("PIR_PLAINTEXT_BITS").unwrap().parse().unwrap();
  let m = parse_exp_to_usize(env::var("PIR_MATRIX_HEIGHT_EXP").unwrap());
  
  CLIFlags {
    m,
    lwe_dim,
    plaintext_bits,
    ele_size,
    num_shards,
  }
}

pub fn parse_exp_to_usize(v: String) -> usize {
  let exp: u32 = v.parse().unwrap();
  2_u32.pow(exp) as usize
}
