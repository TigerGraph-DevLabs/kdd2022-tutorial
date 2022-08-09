## Getting Started

Fill out the [Google Form](https://forms.gle/ncvLTykeFqJWZEU49) to get an invite to a TigerGraph Cloud account. You will be able to provision a database instance and ML Workbench once you follow the invite emailed to you and create an account.

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

#### 4. Create Secret for Graph

Once your graph is created, we can create a secret we will use for authentication with the database. From the previous screen, click on **Admin Portal** in the upper right corner. It will bring you to this screen:

<img src="./img/apUsers.png" alt="drawing" width="800"/>

Under the **Management** tab on the left hand menu bar, select **Users**. Here, you can create a secret. Define an alias and then click the **+** button:
<img src="./img/createSecret.png" alt="drawing" width="800"/>

Make sure to copy the secret that is generated, as you will never see it again.
<img src="./img/copySecret.png" alt="drawing" width="800"/>

Finally, we can paste the secret into the notebook cell below:
<img src="./img/pasteSecret.png" alt="drawing" width="800"/>

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

The code for this tutorial will be in the `kdd2022-tutorial` directory.