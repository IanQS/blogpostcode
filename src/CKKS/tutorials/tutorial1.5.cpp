#include "palisade.h"
#include <iomanip>
#include "../utils/csv_reader.h"
#include "../utils/misc.h"

using namespace lbcrypto;

/**
 * multDepthInvestigation:
 *      Investigate what happens when, ceteris-paribus, the mult-depth is changed
 * @param multDepth : alternate between 1 and 3
 * @param scalingFactorBits: kept at 50
 * @param batchSize: kept at 8
 * @param features: number of features. batchSize should be next power of 2 greater than current
 * @param labels: label of data
 */
void multDepthInvestigation(
    uint multDepth,
    uint scalingFactorBits,
    uint batchSize,
    FeatureMatrixComplex features) {

  CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
      multDepth, scalingFactorBits, batchSize
  );

  std::cout << "\t-Ring dimensionality: " << cc->GetRingDimension() << '\n';

  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();  // the keys we encrypt and decrypt with
  cc->EvalMultKeyGen(keys.secretKey); // our relinearization key. We'll talk about this later

  auto singleFeat = features[0];

  Plaintext pt1 = cc->MakeCKKSPackedPlaintext(singleFeat);

  Ciphertext<DCRTPoly> ct1 = cc->Encrypt(keys.publicKey, pt1);

  /*
   * Exceeding the mult depth in case where multDepth=1 but not in the case of 3
   */
  auto resMult = cc->EvalMult(cc->EvalMult(ct1, ct1), ct1);
  Plaintext pt_res_mult;
  cc->Decrypt(keys.secretKey, resMult, &pt_res_mult);
  pt_res_mult->SetLength(features[0].size());
  std::cout << "\t-Result of taking to 3rd power: " << pt_res_mult << '\n';
}

/**
 * Illustrate the alternative method of getting around the mult-depth issue: ciphertext refreshing under the
 * "interactive" scheme
 * @param multDepth
 * @param scalingFactorBits
 * @param batchSize
 * @param features
 */
void multDepthDecrypt(
    uint multDepth,
    uint scalingFactorBits,
    uint batchSize,
    FeatureMatrixComplex features
) {

  CryptoContext<DCRTPoly>
      cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(multDepth, scalingFactorBits, batchSize, );

  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);

  auto singleFeat = features[0];

  Plaintext pt1 = cc->MakeCKKSPackedPlaintext(singleFeat);

  Ciphertext<DCRTPoly> ct1 = cc->Encrypt(keys.publicKey, pt1);

  /*
   * Exceeding the mult depth in case where multDepth=1 but not in the case of 3
   */

  auto resMultSmall = cc->EvalMult(ct1, ct1);
  Plaintext pt_res_mult1;
  cc->Decrypt(keys.secretKey, resMultSmall, &pt_res_mult1);
  pt_res_mult1->SetLength(features[0].size());
  std::cout << "\t-Squaring and results: " << pt_res_mult1 << '\n';

  auto resMult = cc->EvalMult(pt_res_mult1, ct1);
  Plaintext pt_res_mult;
  cc->Decrypt(keys.secretKey, resMult, &pt_res_mult);
  pt_res_mult->SetLength(features[0].size());
  std::cout << "\t-Result of taking to 3rd power: " << pt_res_mult << '\n';

}

int main() {

  std::cout.precision(3);

  FeatureMatrixComplex features = {{5.7, 3.8, 1.7, 0.3}};
  LabelVector label_vector = {0};
  uint8_t multDepthSmall = 1;
  uint8_t multDepthLarger = 3;


  uint8_t scalingFactorBits = 50;

  uint batchSize = nextPowerOfTwo(features[0].size());

  /*
   * Illustrating mult-depth
   */
  std::cout << std::setw(20) << std::setfill('*') << '\n';
  std::cout
      << "Notice multDepthSmall=1, the result of our taking it to the 3rd power is meaningless. We have gone over the depth"
      << '\n';
  multDepthInvestigation(multDepthSmall, scalingFactorBits, batchSize, features);

  std::cout << std::setw(10) << std::setfill('-') << '\n';

  std::cout
      << "Notice that with a larger mult depth things make sense, the results now make sense"
      << '\n';
  multDepthInvestigation(multDepthLarger, scalingFactorBits, batchSize, features);

  std::cout << std::setw(10) << std::setfill('-') << '\n';

  std::cout
      << "We can avoid the multiplication depth issue by decrpyting and encrypting again as we did in tutorial1"
      << '\n';

  multDepthDecrypt(1, scalingFactorBits, batchSize, features);
}
