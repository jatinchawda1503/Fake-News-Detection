from setuptools import setup,find_packages


setup(name="fakenews",
      version='1.0',
      description="Detects Fake News",
      author='DSA-6',
      packages=find_packages(),
      install_requires = ['pandas==1.4.3','nltk==3.7','scikit-learn==1.1.1','xgboost==1.6.1'],
      zip_safe=False,
      include_package_data=True,
      package_data={'': ['*.csv'],'models':['models/*.joblib']}
      )
