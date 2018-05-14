import java.awt.Toolkit
import java.lang.Thread.sleep

fun main(args: Array<String>) {
    System.out.print("AAA")
    System.out.print("\\7")
    System.out.println("BBB")
    for (i in 1..10) {
        Toolkit.getDefaultToolkit().beep()
        sleep(500)
    }
}