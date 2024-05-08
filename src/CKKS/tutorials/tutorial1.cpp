#include "palisade.h"
#include <iomanip>
#include "../utils/csv_reader.h"

using namespace lbcrypto;

/**
 * learning_w_CKKS:
 *
 * We show how to use CKKS to create a simple GLM. You may find it useful to create a plaintext version alongside
 * this as a sketch before transferring it into CKKS
 *
 * Note: your results may be different from a plaintext version or a version in numpy. This is because CKKS
 * accumulates noise as it does operations on ciphertexts and this noise leads to small errors that accumulate
 * over time
 * @param multDepth
 * @param scalingFactorBits
 * @param batchSize
 * @param features
 * @param labels
 * @param refreshEvery
 * @param verbose
 * @param epochs
 */
void learning_w_CKKS(uint multDepth,
                     uint scalingFactorBits,
                     uint batchSize,
                     FeatureMatrixComplex features,
                     LabelVector labels,
                     int refreshEvery,
                     bool verbose,
                     int epochs
) {

  /////////////////////////////////////////////////////////////////
  //Boilerplate setup
  /////////////////////////////////////////////////////////////////
  CryptoContext<DCRTPoly> cc = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
      multDepth, scalingFactorBits, batchSize
  );

  std::cout << "Ring Dimension: " << cc->GetRingDimension() << '\n';

  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalSumKeyGen(keys.secretKey);


  /////////////////////////////////////////////////////////////////
  //Setup the weights, biases and encrypt all
  /////////////////////////////////////////////////////////////////
  using cMatrix = std::vector<std::complex<double>>;

  cMatrix singleLabelAsVector = {(float) labels[0]};

  // we add a bias to our features which allows us to scale the line of best fit
  // by some offset
  cMatrix biasFeatures = features[0];
  auto it = biasFeatures.begin();
  std::cout << "Vec 0 before adding bias: " << biasFeatures << '\n';
  biasFeatures.insert(it, std::complex<double>(1, 0));
  std::cout << "Vec 0 after adding bias: " << biasFeatures << '\n';

  // the IRIS dataset has 4 features but since we added a bias term
  // above, we generate a vector of 5 elements
  cMatrix weights = {1, 1, -0.1, -1.5, 10.01};
  // Up to now, all the steps we have taken should be familiar to a
  // reader with a machine learning background

  // Below we pack our data into a plaintext. To pack the data means that we
  // encode multiple data points into a single ciphertext. The advantage of
  // this is that it allows us to operate on multiple data at once as opposed
  // to single values of data; think vector operations as opposed to
  // scalar ops in numpy.
  auto singleFeature = cc->MakeCKKSPackedPlaintext(
      biasFeatures
  );
  auto singleLabel = cc->MakeCKKSPackedPlaintext(
      singleLabelAsVector
  );
  auto packedWeights = cc->MakeCKKSPackedPlaintext(weights);

  // Above we packed the values but now we must encrypt them
  auto encFeature = cc->Encrypt(keys.publicKey, singleFeature);
  auto encLabel = cc->Encrypt(keys.publicKey, singleLabel);
  auto encWeights = cc->Encrypt(keys.publicKey, packedWeights);

  double alpha = 0.01;
  Plaintext ptResidual;
  Plaintext ptGradient;
  if (verbose) {
    std::cout << "Label: " << singleLabel << '\n';
  }
  /////////////////////////////////////////////////////////////////
  //Start training
  /////////////////////////////////////////////////////////////////
  for (int i = 1; i < epochs + 1; i++) {
    // offset the index by 1 for some modulus checking below
    if (verbose) {
      std::cout << "Epoch : " << i << '\n';
    }
    // We have two vectors so to accomplish an inner product we
    // first do a hadamard product, and then sum the values
    auto innerProd = cc->EvalSum(
        cc->EvalMult(encFeature, encWeights), batchSize
    );

    // to determine how close we are to the correct value,
    // we find the difference
    auto residual = cc->EvalSub(innerProd, encLabel);

    // The residual variable is a ciphetext that contains our true error
    // value. Thus, we must decrypt it and store the result to a plaintext
    cc->Decrypt(keys.secretKey, residual, &ptResidual);

    // we set the length below because we are only interested in the
    // first value. The remaining elements are 0s (approximately) and are
    // a result of the packing.

    if (verbose) {

      ptResidual->SetLength(1);
      auto decResidual = ptResidual->GetCKKSPackedValue()[0].real();
      auto cost = decResidual * decResidual;

      std::cout << "Residual: " << decResidual << '\t' <<"cost: " << cost << '\n';
    }

    // derivative:
    // J(w) = (Wx - y)
    // /partial J(w) = x.T * (y-Wx)  ( Equivalent to dot prod)

    auto err_partial_w = cc->EvalSum(
        cc->EvalMult(encFeature, residual), batchSize
    );
    // Scale by learning rate
    auto update = cc->EvalMult(alpha, err_partial_w);

    // Since this is meant as a tutorial, we print out the gradient
    // after every training step.
    if (verbose) {
      cc->Decrypt(keys.secretKey, update, &ptGradient);
      ptGradient->SetLength(4);
      std::cout << "Update: ";
      for (auto &v: ptGradient->GetCKKSPackedValue()){
        std::cout << v.real() << ',';
      }
      std::cout << std::endl;
    }
    encWeights = cc->EvalSub(encWeights, update);

    /*
     * Refreshing scheme that we mentioned earlier.
     *  Try it! Fix the other params to
     *    multDepth = 5
     *    scalingFactorBits = 40
     *    batchSize  = 4096
     *
     *  and change the refreshEvery from 2 (success) to 3(failure)
     *
     * What this is doing is refreshing the weights. It's effectively creating a new one and replacing the old
     * one with it. This lets us keep operating on the values without exceeding the noise threshold
     */
    if ((refreshEvery != 0) && (i % refreshEvery == 0)) {
      Plaintext clearWeights;
      cc->Decrypt(keys.secretKey, encWeights, &clearWeights);  // We don't actually use this. We use it to decrpt
      auto oldWeights = clearWeights->GetCKKSPackedValue();
      clearWeights->SetLength(4);
      encWeights = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(oldWeights));

      if (verbose) {
        std::cout << std::setw(20) << std::setfill('*') << '\n';
      }
    } else {
      if (verbose) {
        std::cout << std::setw(20) << std::setfill('*') << '\n';
      }
    }  // ENDIF ((refreshEvery != 0) && (i % refreshEvery == 0))
  }

  std::cout << '\n';
}

int main() {

  std::cout.precision(3);
  FeatureMatrixComplex features = {{5.7, 3.8, 1.7, 0.3}};
  LabelVector labels = {0};
  /*
   * Step 1: Parameter Discussion
   */

  // Depth of multiplication supported:
  // (a * b) + (c * d) has a mult depth of 1
  // a * b * c has a mult depth of 2
  // We may come up on the term "towers" later and multDepthSmall is one less
  // i.e multDepthSmall == (numTowers - 1)
  // Note: there's not really a 1-size-fits-all setup here so you'll need to know
  // your underlying algorithm to use this effectively
  uint8_t multDepthSmall = 5;

  // In the original paper they discuss a scaling factor which they multiply by
  // to prevent rounding errors from destroying the significant figures during encoding
  // Note: this specifies the bit LENGTH of the scaling factor but not the scaling factor
  //itself
  // Note: I personally stick to 50
  uint8_t scalingFactorBits = 40;

  // The number of slots to use. This is typically set to a power of 2 larger than
  // the dimensionality of the data. Our iris dataset has 4 features and the next
  // power of 2 is 8
  int batchSize = 4096;

  // This is an "internal" trick but if you're familiar with CKKS and other encryption schemes like it
  // you'll know they accumulate more and more noise as more operations are carried out. One such "fix"
  // is to decrypt-recrypt the result. This reduces the noise again but the tradeoff is that it is slower

  int refreshEvery = 1;

  bool verbose = true;

  std::cout << std::setw(20) << std::setfill('*') << '\n' << '\n';
  std::cout << "CKKS: Charting the loss " << '\n';
  learning_w_CKKS(multDepthSmall, scalingFactorBits, batchSize, features, labels, refreshEvery, verbose, 20);
  std::cout << std::setw(20) << std::setfill('*') << '\n';

}

