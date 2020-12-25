rm -rf */build/ **/*.so
echo 'rm before'

echo 'build emd '

cd emd
python setup.py build_ext
cp build/lib*/*.so ./

cd ..

echo 'build points collection '

cd points_collection_ops
python setup.py build_ext
cp build/lib*/*.so ./

cd ..
echo 'build scatter feature'

cd scatter_feature_ops
python setup.py build_ext
cp build/lib*/*.so ./

cd ..
