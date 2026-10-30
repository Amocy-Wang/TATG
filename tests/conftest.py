from pathlib import Path


import pytest


@pytest.fixture()
def sample_repo() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture()
def generated_java_project(tmp_path: Path) -> Path:
    """Create a simple multi-class Maven-style Java project for scope tests."""
    root = tmp_path / "generated_java_project"
    (root / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True, exist_ok=True)
    (root / "src" / "main" / "java" / "com" / "acme" / "util").mkdir(parents=True, exist_ok=True)

    (root / "pom.xml").write_text(
        """
<project xmlns=\"http://maven.apache.org/POM/4.0.0\"
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"
         xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd\">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>generated-java-project</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "src" / "main" / "java" / "com" / "acme" / "Calculator.java").write_text(
        """
package com.acme;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int div(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("b cannot be zero");
        }
        return a / b;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "src" / "main" / "java" / "com" / "acme" / "UserService.java").write_text(
        """
package com.acme;

public class UserService {
    public String normalize(String name) {
        return name == null ? "" : name.trim().toLowerCase();
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "src" / "main" / "java" / "com" / "acme" / "util" / "StringUtil.java").write_text(
        """
package com.acme.util;

public class StringUtil {
    public static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture()
def generated_multi_module_project(tmp_path: Path) -> Path:
    """Create a simple Maven multi-module project for scope behavior tests."""
    root = tmp_path / "generated_multi_module_project"
    module_a = root / "module-a"
    module_b = root / "module-b"

    for module in (module_a, module_b):
        (module / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True, exist_ok=True)

    (root / "pom.xml").write_text(
        """
<project xmlns=\"http://maven.apache.org/POM/4.0.0\"
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"
         xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd\">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>multi-module-root</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>pom</packaging>
  <modules>
    <module>module-a</module>
    <module>module-b</module>
  </modules>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (module_a / "pom.xml").write_text(
        """
<project xmlns=\"http://maven.apache.org/POM/4.0.0\"
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"
         xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd\">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.acme</groupId>
    <artifactId>multi-module-root</artifactId>
    <version>1.0-SNAPSHOT</version>
  </parent>
  <artifactId>module-a</artifactId>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (module_b / "pom.xml").write_text(
        """
<project xmlns=\"http://maven.apache.org/POM/4.0.0\"
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"
         xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd\">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.acme</groupId>
    <artifactId>multi-module-root</artifactId>
    <version>1.0-SNAPSHOT</version>
  </parent>
  <artifactId>module-b</artifactId>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (module_a / "src" / "main" / "java" / "com" / "acme" / "BillingService.java").write_text(
        """
package com.acme;

public class BillingService {
    public int total(int price, int tax) {
        return price + tax;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (module_b / "src" / "main" / "java" / "com" / "acme" / "NotificationService.java").write_text(
        """
package com.acme;

public class NotificationService {
    public String channel(boolean urgent) {
        return urgent ? "sms" : "email";
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return root
