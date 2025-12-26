FROM python:3.10-slim-bullseye

# ───────────────────────────────────────────────────────────────────────
# Step 1: Install OS‐level dependencies for Python, R, AND Java/Spark/SBT
# ───────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    r-base \
    r-base-dev \
    openjdk-11-jdk-headless \
    curl \
    wget \
    ca-certificates \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (so Spark and sbt know where Java is)
RUN ln -sf $(dirname $(dirname $(readlink -f $(which java)))) /usr/lib/jvm/java-home
ENV JAVA_HOME=/usr/lib/jvm/java-home

# Test Java setup
RUN echo "Testing Java setup..." && \
    echo "JAVA_HOME is set to: $JAVA_HOME" && \
    ls -l "$JAVA_HOME/bin" && \
    "$JAVA_HOME/bin/java" -version

# ───────────────────────────────────────────────────────────────────────
# Step 2: Install UV and Jupyter
# ───────────────────────────────────────────────────────────────────────
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Jupyter
RUN pip install --no-cache-dir jupyterlab

# ───────────────────────────────────────────────────────────────────────
# Step 2.5: Install Python library using UV
# ───────────────────────────────────────────────────────────────────────
# Install at system level
COPY python_lib /app/python_lib  
WORKDIR /app/python_lib
# note: this will install the project in editable mode, so any changes to the project on the image will be reflected.
# docker image so no need to use uv sync to install to virtual environment
RUN uv pip install --system -e . --group dev

# Caveat: 
# 1. when launching image while mounting python_lib to app folder, the mounted python_lib will override the python_lib in the image, hence the project is live reloaded.
# 2. when launching image without mounting python_lib to app folder, the python_lib in the image will be used, hence the project is using the version when the image was built.

# Switch back to main app directory
WORKDIR /app

# ───────────────────────────────────────────────────────────────────────
# Step 3: Install R kernel and any R packages (optional)
# ───────────────────────────────────────────────────────────────────────
# RUN R -e "install.packages('IRkernel', repos='http://cran.us.r-project.org')" && \
#     R -e "IRkernel::installspec(user = FALSE)"

# COPY install_r_packages.R /tmp/install_r_packages.R
# RUN Rscript /tmp/install_r_packages.R

# ───────────────────────────────────────────────────────────────────────
# Step 4: Install sbt (Scala Build Tool) for Apache Toree
# ───────────────────────────────────────────────────────────────────────
# Add sbt’s official Debian repository key and source, then install sbt
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" > /etc/apt/sources.list.d/sbt.list \
 && echo "deb https://repo.scala-sbt.org/scalasbt/debian /" >> /etc/apt/sources.list.d/sbt.list \
 && curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x99E82A75642AC823" | apt-key add - \
 && apt-get update \
 && apt-get install -y sbt \
 && rm -rf /var/lib/apt/lists/*

# ───────────────────────────────────────────────────────────────────────
# Step 5: Download & extract Apache Spark
#   * Version is synced with scala_lib/gradle.properties via Makefile
# ───────────────────────────────────────────────────────────────────────
ARG SPARK_VERSION=3.4.1
ARG HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

# Print versions for verification
RUN echo "=== Building with Spark ${SPARK_VERSION} and Hadoop ${HADOOP_VERSION} ==="

RUN wget --quiet https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
 && mkdir -p /opt \
 && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt \
 && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME \
 && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# ───────────────────────────────────────────────────────────────────────
# Step 6: Install Apache Toree via pip, then register the Scala kernel
# ───────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir toree \
 && jupyter toree install \
      --spark_home=$SPARK_HOME \
      --interpreters=Scala \
      --kernel_name="apache_toree_scala" \
      --sys-prefix

# ───────────────────────────────────────────────────────────────────────
# Step 7: Expose Jupyter port, set working directory, and entrypoint
# ───────────────────────────────────────────────────────────────────────
EXPOSE 8888

# Set working directory
WORKDIR /app

# Start Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
