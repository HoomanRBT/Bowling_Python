-- MySQL dump 10.13  Distrib 8.0.16, for Win64 (x86_64)
--
-- Host: localhost    Database: configuration
-- ------------------------------------------------------
-- Server version	8.0.16

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
 SET NAMES utf8 ;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ip`
--

DROP TABLE IF EXISTS `ip`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `ip` (
  `idip` int(11) NOT NULL AUTO_INCREMENT,
  `device_name` varchar(15) DEFAULT NULL,
  `ip_address` varchar(15) DEFAULT NULL,
  `username` varchar(20) DEFAULT NULL,
  `passwd` varchar(20) DEFAULT NULL,
  `lane` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`idip`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ip`
--

LOCK TABLES `ip` WRITE;
/*!40000 ALTER TABLE `ip` DISABLE KEYS */;
INSERT INTO `ip` VALUES (1,'server','192.168.1.1',NULL,NULL,NULL),(2,'cam1','192.168.1.2','admin','12345','1,2'),(3,'cam2','192.168.1.3','admin','12345','3,4'),(4,'cam3','192.168.1.4','admin','admin','5,6'),(5,'cam4','192.168.1.5','root','admin','7,8'),(6,'cam5','192.168.1.6','admin','admin','9'),(7,'raspberry1','192.168.1.7','raspberry1','abc1','1'),(8,'raspberry2','192.168.1.8','raspberry2','abc2','2'),(9,'raspberry3','192.168.1.9','raspberry3','abc3','3'),(10,'raspberry4','192.168.1.10','raspberry4','abc4','4'),(11,'raspberry5','192.168.1.11','raspberry5','abc5','5'),(12,'raspberry6','192.168.1.12','raspberry6','abc6','6'),(13,'raspberry7','192.168.1.13','raspberry7','abc7','7'),(14,'raspberry8','192.168.1.14','raspberry8','abc8','8'),(15,'raspberry9','192.168.1.15','raspberry9','abc9','9'),(16,'raspberry10','192.168.1.16','raspberry10','abc10','10');
/*!40000 ALTER TABLE `ip` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-07-28  0:23:47
