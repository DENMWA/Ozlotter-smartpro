# Deployment Guide

## Render Deployment Options

### Option 1: Using render.yaml (Infrastructure-as-Code)
The repository includes a `render.yaml` file for automatic deployment configuration.

**Advantages:**
- Version controlled deployment configuration
- Automatic deployment on git push
- Consistent across environments

**Configuration:**
- Build Command: `pip install -r enhanced_requirements.txt`
- Start Command: `streamlit run enhanced_streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

### Option 2: Manual Render Dashboard Configuration
If you prefer manual configuration or want to override render.yaml:

1. **Create New Web Service** in Render dashboard
2. **Connect Repository**: DENMWA/Ozlotter-smartpro
3. **Branch**: main
4. **Build Command**: `pip install -r enhanced_requirements.txt`
5. **Start Command**: `streamlit run enhanced_streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
6. **Environment Variables** (optional):
   - `PYTHON_VERSION`: 3.10
   - `STREAMLIT_SERVER_HEADLESS`: true

### Troubleshooting Common Issues

#### "ModuleNotFoundError: No module named 'plotly'"
- **Cause**: Using `requirements.txt` instead of `enhanced_requirements.txt`
- **Fix**: Update Build Command to use `enhanced_requirements.txt`

#### "No such option: -r" Error
- **Cause**: Build and Start commands are mixed together
- **Fix**: Separate Build Command and Start Command in Render settings

#### App loads but missing enhanced features
- **Cause**: Using `streamlit_app.py` instead of `enhanced_streamlit_app.py`
- **Fix**: Update Start Command to use `enhanced_streamlit_app.py`

#### Render Cache Issues
If changes don't take effect after updating configuration:
1. Go to Render dashboard → Your service → Settings
2. Click "Manual Deploy"
3. Select "Clear build cache"
4. Deploy again

#### Python Version Compatibility
- Enhanced version requires Python 3.10+ for TensorFlow compatibility
- Original version works with Python 3.8+

### Performance Recommendations
- Use **Standard** instance type for enhanced version (TensorFlow requires more memory)
- Use **Starter** instance type for original version
- Enable auto-deploy for continuous deployment

### Alternative Deployment Methods

#### Deploy Original Version Only
If you want to deploy the simpler original version without AI features:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

#### Deploy with Reduced Dependencies
If TensorFlow causes memory issues, you can temporarily disable neural networks:
1. Comment out TensorFlow in `enhanced_requirements.txt`
2. Set `enable_neural=False` by default in the enhanced app
3. Use Standard instance type

### Verification Steps
After deployment, verify:
1. ✅ App loads without import errors
2. ✅ Enhanced dashboard displays correctly
3. ✅ All AI prediction methods are available
4. ✅ Plotly charts render properly
5. ✅ Data fetching works correctly

### Support
If you continue experiencing issues:
1. Check Render build logs for specific error messages
2. Verify your Render service is using the correct branch (main)
3. Ensure render.yaml changes are deployed
4. Try manual dashboard configuration as fallback
