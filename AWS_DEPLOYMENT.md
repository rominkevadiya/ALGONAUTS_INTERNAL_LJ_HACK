# AWS Deployment Guide (Jury & Friends)

Since this is a quick deployment to show the jury and your friends, we will use a single **AWS EC2 Instance**. This approach is simple, requires no load balancer, and gets your app running quickly.

## 1. Launch an EC2 Instance

1. Log in to the [AWS Management Console](https://console.aws.amazon.com/ec2/).
2. Navigate to **EC2** and click **Launch instance**.
3. **Name**: `algonauts-streamlit-app` (or whatever you prefer).
4. **OS (AMI)**: Select **Ubuntu** (Ubuntu Server 24.04 LTS is a good choice).
5. **Instance Type**: 
   > [!IMPORTANT]
   > Your app uses `torch`, `opencv`, and `google-genai`. A free-tier `t2.micro` (1GB RAM) will likely crash. Please select **`t3.medium`** (4GB RAM) or **`t3.large`** (8GB RAM) for stability during the jury review.
6. **Key Pair**: Create a new key pair (RSA, `.pem`), download it, and keep it safe. You will need it to connect to the instance.
7. **Network Settings**:
   - Check **Allow SSH traffic from Anywhere**.
   - Check **Allow HTTP traffic from the internet**.
   - **Important**: We need to open port 8501 for Streamlit.
     - Under "Network Settings", click **Edit**.
     - Add a **Custom TCP rule**: Port range `8501`, Source type `Anywhere` (`0.0.0.0/0`).
8. **Storage**: Increase the default 8GB to at least **20GB** or **30GB** because PyTorch is large.
9. Click **Launch instance**.

## 2. Connect and Transfer Files

Once the instance is running, you need to get your code onto the server.

### Option A: Using Git (Recommended)
If your code is on GitHub (public or private):
Connect to your EC2 instance via SSH:
```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```
Then clone your repo:
```bash
git clone <your-repo-url> app-repo
cd app-repo
```

### Option B: Copying files directly from your PC
If you don't use Git, you can copy the files from your Windows machine using SCP or an SFTP client like Cyberduck/FileZilla.
Open PowerShell on your PC and run:
```powershell
scp -i "path\to\your-key.pem" -r "d:\ALGONAUTS_INTERNAL_LJ_HACK-main" ubuntu@<your-ec2-public-ip>:~/app-repo
```

> [!WARNING]
> Don't forget your `.env` file! If it is gitignored, you will need to create it manually on the EC2 instance or copy it over.

## 3. Setup and Run the App

I have created a setup script for you in the project root: [`setup_ec2.sh`](file:///d:/ALGONAUTS_INTERNAL_LJ_HACK-main/setup_ec2.sh). 

Connect to your EC2 instance via SSH, navigate to the project directory, and run:

```bash
# Make the script executable
chmod +x setup_ec2.sh

# Run the setup script (installs python, requirements, etc.)
./setup_ec2.sh
```

## 4. Keep the App Running

If you run `streamlit run` normally, it will close when you disconnect from the server. To keep it running for the jury:

1. Create a virtual terminal session using `tmux`:
   ```bash
   tmux new -s myapp
   ```
2. Activate your environment and start the app:
   ```bash
   source .venv/bin/activate
   streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
   ```
3. **Detach** from the session so it stays running in the background:
   - Press `Ctrl + B`, then release both and press `D`.
4. You can now close your SSH connection!

## 5. View your App!

Open your browser and go to:
`http://<your-ec2-public-ip>:8501`

*(Share this link with the jury and your friends!)*

> [!NOTE]
> When you are done with the hackathon, remember to go to the AWS console and **Stop** or **Terminate** your EC2 instance so you don't keep getting charged for the `t3.medium` instance!
