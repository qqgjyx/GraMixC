using SGtSNEpi, MLDatasets, ImageFeatures, Random, Images, MAT
using CairoMakie, Colors, Makie
X_train, L_train = MNIST(Float64, split=:train)[:];
X_test, L_test = MNIST(Float64, split=:test)[:];
X = cat(X_train, X_test, dims=3);
L = cat(L_train, L_test, dims=1);
# L = vec(Int.(matread("../out/MNIST/predictions.mat")["predictions"]))

n = size( X, 3 );
X = permutedims( X, [2, 1, 3] );

F = zeros( n, 324 );
for img = 1:n
    F[img,:] = create_descriptor( X[:,:,img], HOG(; cell_size = 7) )
  end

Random.seed!(0);
Y0 = 0.01 * randn( n, 2 );

A = pointcloud2graph(F, 5, 15) # perplexity = 5, k = 15
Y = sgtsnepi(A; Y0 = Y0);


show_embedding(Y, L; res = (2000, 2000), mrk_size = 4)

# show the UMAP embedding
using UMAP
Y_umap = umap(F'; n_neighbors=15, min_dist=0.1, n_epochs=200)
show_embedding(Y_umap', L; res = (2000, 2000), mrk_size = 4)


# show the t-SNE embedding
# Note: Standard t-SNE can be computationally intensive for large datasets
# because it computes pairwise distances between all points (O(n²) complexity)
# compared to SGtSNEpi which uses a grid-based approximation for repulsive forces
using TSne
A_dense = Matrix(A);
for i in 1:size(A_dense, 1)
    A_dense[i, i] = 0;
end
Y_tsne = tsne(A_dense, 2, 250, 250, 5.0; verbose=true);
show_embedding(Y_tsne, L; res = (2000, 2000), mrk_size = 4)

# show the PCA embedding
using MultivariateStats  # PCA is in MultivariateStats package
M = fit(PCA, F'; maxoutdim=2)
Y_pca = transform(M, F')
show_embedding(Y_pca', L; res = (2000, 2000), mrk_size = 4)


