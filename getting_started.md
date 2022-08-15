## Getting Started

### Provisioning a Database Instance

First, we will have to provision a TigerGraph Cloud instance. Once you follow the invite link in the email you recieve from the Google Form, you will see a page like the below:

<img src="./img/tgCreateSolution.png" alt="drawing" width="800"/>

Click **Create Solution** in the upper right hand corner. You will then see:

<img src="./img/blankInstance.png" alt="drawing" width="800"/>

Select **Blank v3.6.1** and scroll to the bottom of the page to continue. This will then bring you to the instance configuration page.

<img src="./img/instanceConfig1.png" alt="drawing" width="800"/>

Select **AWS** as the platform, **N. Virginia** as the region, a **Public** endpoint, and the **TG.C8.M32** Instance Type. Leave the defaults below, scroll to the bottom of the page, and click **Next**

<img src="./img/instanceConfig2.png" alt="drawing" width="800"/>

We will then name and tag the solution we are provisioning. **The names, tags, and subdomains must be unique, so choose something that is identifiable to you**.

<img src="./img/instanceName.png" alt="drawing" width="800"/>

After clicking next, you should see a confirmation page where you can check the details and then hit **Submit**.

<img src="./img/instanceConfirm.png" alt="drawing" width="800"/>

This will then take you back to the solutions page, where you can find your solution after a few minutes:

<img src="./img/mySolutionsPage.png" alt="drawing" width="800"/>


#### 2. Connect to GraphStudio

Once your solution is provisioned, we want to connect to the GraphStudio UI. To do this, click on the **Applications** icon and select GraphStudio.

<img src="./img/mySolutionsGS.png" alt="drawing" width="800"/>


#### 3. Create Graph

This will take you to the GraphStudio landing page, where we can create our graph. Click on the **Global View** button and select **Create a graph**:

<img src="./img/createGraphGS.png" alt="drawing" width="800"/>

Clicking this will bring up the following pop-up. Fill in the graph name as `KDD_2022_NFT`, and then select **Create**:

<img src="./img/createGraph.png" alt="drawing" width="800"/>

After this, you should see the following:

<img src="./img/designSchema.png" alt="drawing" width="800"/>

### Using ML Workbench
We will use the ML Workbench to perform data analysis and machine learning on the graph data. To do this, we will provision a notebook server with **4 CPU cores and 16 GB of RAM**.
To start, view the **Solutions** page on TigerGraph Cloud:

<img src="./img/tgCloudSolutions.png" alt="drawing" width="800"/>

In the upper left hand corner, click on the **Tools** tab. This will bring you to the following screen:

<img src="./img/mlwbTools.png" alt="drawing" width="800"/>

From there, click on the **ML Workbench** button:

<img src="./img/mlwbHomepage.png" alt="drawing" width="800"/>

Once you are on the MLWB homepage, click on **Notebooks** in the left hand menu bar:

<img src="./img/mlwbNewNB.png" alt="drawing" width="800"/>

Click on the **New Notebook** button:

<img src="./img/mlwbNBCreation.png" alt="drawing" width="800"/>

Name your notebook, select the **tigergraphml/kf-pytorch:kdd2022** image, and use 4 CPU cores and 16 GB of RAM. Scroll to the bottom of the page and click **Launch**:

<img src="./img/mlwbLaunchNB.png" alt="drawing" width="800"/>

This will take you back to the MLWB notebooks homepage. Click on **Connect** to connect to the notebook server:

<img src="./img/mlwbNBConnect.png" alt="drawing" width="800"/>

This will take you to the notebook homepage, seen here:

<img src="./img/mlwbNBHome.png" alt="drawing" width="800"/>

The code for this tutorial will be in the `kdd2022-tutorial` directory. On the left hand side, click on the **Files** tab, then select the `kdd2022-tutorial` directory, and within that select `notebooks`. Click on `0-load_data.ipynb` to open the first notebook. You should see something like this:

<img src="./img/mlwbNB0.png" alt="drawing" width="800"/>

In order to connect to your TigerGraph database, we will be using the connection tool on the left hand tab of Jupyter. Click the TigerGraph logo to see a screen similar to this:

<img src="./img/solutionConnectionTab.png" alt="drawing" width="800"/>

**Note:** If you do not see any solutions, refresh the notebook page and try again.

Select your solution and click **Connect**:

<img src="./img/solutionConnection.png" alt="drawing" width="800"/>

Copy the code from the pop-up and paste it into the first code cell. This will create the connection to the TigerGraph database.

<img src="./img/pastedSolutionDetails.png" alt="drawing" width="800"/>

You will follow the same connection process for the other notebooks.